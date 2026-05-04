"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            print(f"Loading reranker model: {self.model_name}...")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
            
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        
        results = []
        for score, doc in zip(scores, documents):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=0
            ))
            
        # Sắp xếp giảm dần theo điểm rerank
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Cập nhật thứ hạng và lấy top_k
        final_results = results[:top_k]
        for i, res in enumerate(final_results):
            res.rank = i + 1
            
        return final_results


class FlashrankReranker:
    """Lightweight alternative (<5ms)."""
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        from flashrank import Ranker, RerankRequest
        if self._model is None:
            self._model = Ranker(model_name=self.model_name, cache_dir="/tmp")
            
        passages = [{"id": i, "text": d["text"], "meta": d.get("metadata", {})} for i, d in enumerate(documents)]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        
        reranked = []
        for i, res in enumerate(results[:top_k]):
            reranked.append(RerankResult(
                text=res["text"],
                original_score=0.0, # Flashrank doesn't return original
                rerank_score=float(res["score"]),
                metadata=res.get("meta", {}),
                rank=i + 1
            ))
        return reranked


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    import statistics
    times = []
    # Warmup
    reranker.rerank(query, documents)
    
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)  # ms
        
    return {
        "avg_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times)
    }



if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
