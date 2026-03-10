"""
eval_rag/embeddings/embedder.py
Upgraded: nlpaueb/legal-bert-base-uncased (same as opponent RAG)
"""
from sentence_transformers import SentenceTransformer

# ✅ Upgraded from BAAI/bge-base-en-v1.5 → Legal-BERT
# Purpose-trained on legal corpora — better for Pakistani law context
_model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")

def embed(text: str) -> list:
    """Ingestion-time embedding."""
    return _model.encode(text, normalize_embeddings=True).tolist()

def embed_query(text: str) -> list:
    """Query-time embedding with legal context prefix."""
    prefixed = f"legal query: {text}"
    return _model.encode(prefixed, normalize_embeddings=True).tolist()

def embed_fn(texts: list) -> list:
    """Batch embedding — matches opponent RAG interface."""
    return _model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()