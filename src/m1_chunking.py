"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all documents from data/ and convert to markdown using MarkItDown."""
    from markitdown import MarkItDown
    md = MarkItDown()
    docs = []
    # Chỉ xử lý các file văn bản sạch (md, txt)
    extensions = ["*.md", "*.txt"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    print(f"Found {len(files)} files to process...")
    for fp in sorted(files):
        if not os.path.exists(fp): continue


        try:
            print(f"Converting {os.path.basename(fp)}...")
            result = md.convert(fp)
            docs.append({
                "text": result.text_content, 
                "metadata": {
                    "source": os.path.basename(fp),
                    "extension": os.path.splitext(fp)[1]
                }
            })
        except Exception as e:
            print(f"Error converting {fp}: {e}")
            # Fallback for .md files if markitdown fails
            if fp.endswith(".md"):
                with open(fp, encoding="utf-8") as f:
                    docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks), "strategy": "basic"}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks), "strategy": "basic"}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    """
    metadata = metadata or {}
    import re
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    # 1. Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', text) if s.strip()]
    if not sentences:
        return []

    # 2. Encode sentences
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)

    # 3. Group sentences by similarity
    chunks = []
    current_group = [sentences[0]]
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            # Tạo chunk mới
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])
            
    # Thêm group cuối cùng
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))
        
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    
    # 1. Tạo parents (dùng logic tương tự basic nhưng size lớn hơn)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents = []
    current_parent_text = ""
    
    for para in paragraphs:
        if len(current_parent_text) + len(para) > parent_size and current_parent_text:
            p_idx = len(parents)
            pid = f"p_{metadata.get('source', 'doc')}_{p_idx}"
            parents.append(Chunk(
                text=current_parent_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))
            current_parent_text = ""
        current_parent_text += para + "\n\n"
        
    if current_parent_text.strip():
        p_idx = len(parents)
        pid = f"p_{metadata.get('source', 'doc')}_{p_idx}"
        parents.append(Chunk(
            text=current_parent_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))

    # 2. Chia mỗi parent thành các con (children)
    children = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        p_text = parent.text
        # Slide window không overlap đơn giản
        for i in range(0, len(p_text), child_size):
            child_text = p_text[i:i + child_size].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                    parent_id=pid
                ))
                
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    """
    metadata = metadata or {}
    import re
    
    # 1. Tách theo headers (# ## ###)
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    
    chunks = []
    current_header = "Intro"
    current_content = ""
    
    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            if current_content.strip():
                chunks.append(Chunk(
                    text=f"{current_header}\n{current_content}".strip(),
                    metadata={**metadata, "section": current_header, "strategy": "structure"}
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
            
    # Group cuối cùng
    if current_content.strip():
        chunks.append(Chunk(
            text=f"{current_header}\n{current_content}".strip(),
            metadata={**metadata, "section": current_header, "strategy": "structure"}
        ))
        
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Chạy tất cả các chiến thuật và so sánh thống kê.
    """
    results = {
        "basic": {"count": 0, "lengths": []},
        "semantic": {"count": 0, "lengths": []},
        "hierarchical": {"parents": 0, "children": 0, "child_lengths": []},
        "structure": {"count": 0, "lengths": []}
    }
    
    for doc in documents:
        text = doc["text"]
        meta = doc["metadata"]
        
        # 1. Basic
        b_chunks = chunk_basic(text, metadata=meta)
        results["basic"]["count"] += len(b_chunks)
        results["basic"]["lengths"].extend([len(c.text) for c in b_chunks])
        
        # 2. Semantic
        s_chunks = chunk_semantic(text, metadata=meta)
        results["semantic"]["count"] += len(s_chunks)
        results["semantic"]["lengths"].extend([len(c.text) for c in s_chunks])
        
        # 3. Hierarchical
        h_parents, h_children = chunk_hierarchical(text, metadata=meta)
        results["hierarchical"]["parents"] += len(h_parents)
        results["hierarchical"]["children"] += len(h_children)
        results["hierarchical"]["child_lengths"].extend([len(c.text) for c in h_children])
        
        # 4. Structure
        st_chunks = chunk_structure_aware(text, metadata=meta)
        results["structure"]["count"] += len(st_chunks)
        results["structure"]["lengths"].extend([len(c.text) for c in st_chunks])

    # In bảng so sánh
    print("\n" + "="*60)
    print(f"{'Strategy':<15} | {'Chunks':<10} | {'Avg Len':<10} | {'Min':<6} | {'Max':<6}")
    print("-" * 60)
    
    final_stats = {}
    for name, data in results.items():
        if name == "hierarchical":
            lens = data["child_lengths"]
            count_str = f"{data['parents']}p/{data['children']}c"
        else:
            lens = data["lengths"]
            count_str = str(data["count"])
            
        if lens:
            avg = sum(lens) / len(lens)
            mi, ma = min(lens), max(lens)
        else:
            avg, mi, ma = 0, 0, 0
            
        print(f"{name:<15} | {count_str:<10} | {avg:<10.1f} | {mi:<6} | {ma:<6}")
        final_stats[name] = {"count": count_str, "avg": avg, "min": mi, "max": ma}
        
    return final_stats


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        print("No documents found! Check DATA_DIR or file extensions.")
    else:
        print(f"Loaded {len(docs)} documents")
        results = compare_strategies(docs)



if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
