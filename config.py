"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# --- LLM Models ---
# GEMINI_MODEL = "gemini-2.5-flash"
LLM_MODEL = "gpt-4o-mini"
GEMINI_MODEL = LLM_MODEL   # Alias for compatibility

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "embed-multilingual-v3.0"
EMBEDDING_DIM = 1024
GEMINI_EVAL_EMBEDDING = "models/embedding-001"

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 800
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 40
DENSE_TOP_K = 40
HYBRID_TOP_K = 40
RERANK_TOP_K = 5

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
