"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline():
    """Build production RAG pipeline with Parent-Child retrieval."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE (Parent-Child Strategy)")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    docs = load_documents()
    all_chunks = []
    parent_map = {} # Map parent_id -> parent_text
    
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        # Lưu trữ text của parent
        for p in parents:
            parent_map[p.metadata["parent_id"]] = p.text
            
        for child in children:
            all_chunks.append({
                "text": child.text, 
                "metadata": {**child.metadata, "parent_id": child.parent_id}
            })
    print(f"  {len(all_chunks)} child chunks indexed. {len(parent_map)} parents stored.")

    # Step 2: Index (M2)
    search = HybridSearch()
    search.index(all_chunks)

    # Step 3: Reranker (M3)
    reranker = CrossEncoderReranker()

    return search, reranker, parent_map


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker, parent_map: dict = None) -> tuple[str, list[str]]:
    """Run single query with Parent retrieval fallback."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    
    # KỸ THUẬT PRODUCTION: Lấy text của Parent thay vì Child
    contexts = []
    for r in reranked:
        pid = r.metadata.get("parent_id")
        if parent_map and pid in parent_map:
            # Nếu tìm thấy Parent, lấy text của Parent (đầy đủ hơn)
            contexts.append(parent_map[pid])
        else:
            contexts.append(r.text)
    
    # Loại bỏ trùng lặp nếu nhiều child thuộc cùng 1 parent
    unique_contexts = []
    for c in contexts:
        if c not in unique_contexts:
            unique_contexts.append(c)
    contexts = unique_contexts[:3] # Lấy top 3 parent lớn là đủ

    
    # Debug: In ra các đoạn context tìm được
    snippets = [c[:40].replace("\n", " ") + "..." for c in contexts]
    print(f"  🔍 Retrieved {len(contexts)} contexts. Snippets: {snippets}")



    # Step: LLM Generation (Supports both Gemini and OpenAI)
    from config import GOOGLE_API_KEY, OPENAI_API_KEY, GEMINI_MODEL
    
    if "gpt" in GEMINI_MODEL.lower():
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY:
            print("  ⚠️  OPENAI_API_KEY missing — using raw context as answer")
            return contexts[0] if contexts else "Không tìm thấy thông tin.", contexts
        llm = ChatOpenAI(
            model=GEMINI_MODEL, 
            api_key=OPENAI_API_KEY, 
            temperature=0, # Fixed to 0
            max_retries=1
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not GOOGLE_API_KEY:
            print("  ⚠️  GOOGLE_API_KEY missing — using raw context as answer")
            return contexts[0] if contexts else "Không tìm thấy thông tin.", contexts
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, 
            google_api_key=GOOGLE_API_KEY, 
            temperature=0, # Fixed to 0
            max_retries=1
        )



    context_str = "\n\n".join(contexts)
    prompt = (
        "Bạn là một trợ lý AI hữu ích. Trả lời câu hỏi CHỈ dựa trên ngữ cảnh được cung cấp. "
        "Nếu ngữ cảnh không chứa câu trả lời, hãy nói 'Không tìm thấy thông tin trong tài liệu.'\n\n"
        f"Ngữ cảnh:\n{context_str}\n\n"
        f"Câu hỏi: {query}"
    )
    
    try:
        resp = llm.invoke(prompt)
        answer = resp.content
    except Exception as e:
        print(f"  ⚠️  LLM Error: {e}")
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."

        
    return answer, contexts



def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, parent_map: dict):
    """Run evaluation on test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker, parent_map)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker, parent_map = build_pipeline()
    evaluate_pipeline(search, reranker, parent_map)
    print(f"\nTotal: {time.time() - start:.1f}s")
