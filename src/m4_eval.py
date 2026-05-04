"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

from ragas.llms import LangchainLLMWrapper
from config import OPENAI_API_KEY
import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, GOOGLE_API_KEY, GEMINI_MODEL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation using Gemini."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    # 0. Cấu hình LLM và Embedding cho RAGAS

    if "gpt" in GEMINI_MODEL.lower():
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        llm = ChatOpenAI(
            model=GEMINI_MODEL, 
            api_key=OPENAI_API_KEY, 
            temperature=0,
            max_retries=1  # Giới hạn thử lại để tiết kiệm quota
        )
        emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, 
            google_api_key=GOOGLE_API_KEY,
            timeout=120,
            temperature=0, # Added temperature=0
            max_retries=1  # Giới hạn thử lại để tiết kiệm quota
        )

        emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)



    
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(emb)
    
    # 1. Chuẩn bị dữ liệu cho RAGAS
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    # 2. Chạy đánh giá
    print(f"Running RAGAS evaluation using {GEMINI_MODEL}...")
    result = evaluate(
        dataset, 
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb
    )

    # 3. Chuyển đổi kết quả sang định dạng EvalResult
    df = result.to_pandas()
    per_question = []
    
    # Ragas có thể trả về tên cột khác nhau tùy phiên bản (vd: question vs user_input)
    col_map = {
        "question": ["question", "user_input", "query"],
        "answer": ["answer", "response", "result"],
        "contexts": ["contexts", "retrieved_contexts"],
        "ground_truth": ["ground_truth", "reference", "target"]
    }
    
    def get_col(row, key):
        for possible_name in col_map[key]:
            if possible_name in row:
                return row[possible_name]
        return ""

    for _, row in df.iterrows():
        per_question.append(EvalResult(
            question=get_col(row, "question"),
            answer=get_col(row, "answer"),
            contexts=get_col(row, "contexts") if isinstance(get_col(row, "contexts"), list) else [],
            ground_truth=get_col(row, "ground_truth"),
            faithfulness=float(row.get("faithfulness", 0)),
            answer_relevancy=float(row.get("answer_relevancy", 0)),
            context_precision=float(row.get("context_precision", 0)),
            context_recall=float(row.get("context_recall", 0))
        ))

        
    # 3. Tổng hợp kết quả bằng Pandas (cách ổn định nhất)
    summary = df.mean(numeric_only=True)
    
    return {
        "faithfulness": float(summary.get("faithfulness", 0)),
        "answer_relevancy": float(summary.get("answer_relevancy", 0)),
        "context_precision": float(summary.get("context_precision", 0)),
        "context_recall": float(summary.get("context_recall", 0)),
        "per_question": per_question
    }





def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    import statistics
    
    if not eval_results:
        return []

    # 1. Tính điểm trung bình cho mỗi câu hỏi
    scored_results = []
    for res in eval_results:
        scores = [res.faithfulness, res.answer_relevancy, res.context_precision, res.context_recall]
        avg = statistics.mean(scores)
        scored_results.append((avg, res))
        
    # 2. Sắp xếp theo điểm trung bình tăng dần và lấy bottom_n
    scored_results.sort(key=lambda x: x[0])
    bottom = scored_results[:bottom_n]
    
    failures = []
    for avg, res in bottom:
        # 3. Tìm metric tệ nhất
        metrics = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall
        }
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        
        # 4. Chẩn đoán lỗi
        diagnosis = "Unknown issue"
        fix = "Investigate further"
        
        if worst_metric == "faithfulness" and worst_score < 0.85:
            diagnosis = "LLM hallucinating"
            fix = "Tighten prompt, lower temperature, or provide better context"
        elif worst_metric == "context_recall" and worst_score < 0.75:
            diagnosis = "Missing relevant chunks"
            fix = "Improve chunking strategy or optimize retrieval (BM25 + Dense Search)"
        elif worst_metric == "context_precision" and worst_score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            fix = "Add reranking step or improve search ranking"
        elif worst_metric == "answer_relevancy" and worst_score < 0.80:
            diagnosis = "Answer doesn't match question"
            fix = "Improve prompt template or refine question rewriting"
            
        failures.append({
            "question": res.question,
            "worst_metric": worst_metric,
            "score": float(worst_score),
            "diagnosis": diagnosis,
            "suggested_fix": fix
        })
        
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
