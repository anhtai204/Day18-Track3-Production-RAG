"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K, COHERE_API_KEY)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text and handle compound words for better matching."""
    import re
    text = re.sub(r'[.,!?:]', ' ', text.lower())
    
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        refined = []
        for word in segmented.split():
            if "_" in word:
                refined.append(word)
                refined.append(word.replace("_", " "))
            else:
                refined.append(word)
        return " ".join(refined)
    except Exception:
        return text




class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(c["text"]).lower().split() for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.bm25:
            return []
        tokenized_query = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append(SearchResult(
                    text=self.documents[i]["text"],
                    score=float(scores[i]),
                    metadata=self.documents[i]["metadata"],
                    method="bm25"
                ))
        return results


class DenseSearch:    
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            self.client.get_collections()
            self.is_available = True
            print(f"  ✅ Connected to Qdrant Docker at {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception:
            try:
                self.client = QdrantClient(":memory:")
                self.is_available = True
                print("  ✅ Qdrant initialized in :memory: mode (Docker not found).")
            except Exception as e:
                print(f"  ⚠️  Qdrant initialization error: {e}")
                self.is_available = False

        
        self._encoder = None
        self._co = None
        
        # Quyết định dùng Cohere hay Local model
        if "embed" in EMBEDDING_MODEL.lower() and COHERE_API_KEY:
            import cohere
            self._co = cohere.Client(api_key=COHERE_API_KEY)
            print(f"  Using Cohere Cloud: {EMBEDDING_MODEL}")
        else:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            print(f"  Using Local Transformer: {EMBEDDING_MODEL}")

    def _get_vectors(self, texts: list[str], input_type: str = "search_document"):
        """Lấy embeddings từ Cohere hoặc Local model."""
        if self._co:
            try:
                response = self._co.embed(
                    texts=texts,
                    model=EMBEDDING_MODEL,
                    input_type=input_type,
                    embedding_types=["float"]
                )
                return response.embeddings.float
            except Exception as e:
                print(f"  ⚠️  Cohere Error: {e}")
                return None
        else:
            vectors = self._encoder.encode(texts, show_progress_bar=False)
            return vectors.tolist()

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Create collection and index chunks."""
        if not self.is_available or not chunks:
            return
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        try:
            # Xóa collection cũ để tránh trùng lặp dữ liệu
            try:
                self.client.delete_collection(collection_name=collection)
                print(f"  🗑️ Deleted old collection: {collection}")
            except Exception:
                pass

            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            
            texts = [c["text"] for c in chunks]
            vectors = self._get_vectors(texts, input_type="search_document")
            if vectors is None: return

            points = []
            for i, (v, c) in enumerate(zip(vectors, chunks)):
                points.append(PointStruct(
                    id=i, 
                    vector=v, 
                    payload={**c["metadata"], "text": c["text"]}
                ))
            
            self.client.upsert(collection, points)
        except Exception as e:
            print(f"  ⚠️  Dense Indexing Error: {e}")
            self.is_available = False

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not self.is_available: return []
        
        try:
            query_vectors = self._get_vectors([query], input_type="search_query")
            if not query_vectors: return []
            query_vector = query_vectors[0]

            if hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=collection, 
                    query_vector=query_vector, 
                    limit=top_k
                )
            else:
                from qdrant_client.models import QueryRequest
                response = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=top_k
                )
                hits = response.points

            
            return [SearchResult(
                text=hit.payload["text"], 
                score=hit.score, 
                metadata=hit.payload, 
                method="dense"
            ) for hit in hits]
        except Exception as e:
            print(f"  ⚠️  Dense Search Error: {e}")
            return []




def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF."""
    rrf_scores = {} # text -> {"score": float, "result": SearchResult}
    
    for result_list in results_list:
        for rank, res in enumerate(result_list):
            if res.text not in rrf_scores:
                rrf_scores[res.text] = {"score": 0.0, "result": res}
            rrf_scores[res.text]["score"] += 1.0 / (k + rank + 1)
            
    # Sắp xếp và trả về top_k
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    final_results = []
    for item in sorted_items:
        res = item["result"]
        final_results.append(SearchResult(
            text=res.text,
            score=item["score"],
            metadata=res.metadata,
            method="hybrid"
        ))
    return final_results



class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
