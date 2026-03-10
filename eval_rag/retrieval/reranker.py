"""
eval_rag/retrieval/reranker.py
Upgraded: Lazy-loaded class with graceful fallback (matches opponent RAG pattern)
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker — lazy loaded to avoid slow startup.
    Falls back gracefully if model unavailable.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Reranker loaded: {self.model_name}")
            return True
        except Exception as e:
            logger.warning(f"Reranker unavailable: {e}. Falling back to hybrid scores.")
            return False

    def available(self) -> bool:
        return self._ensure_model()

    def score_pairs(self, query: str, docs: List[str]) -> Optional[List[float]]:
        if not self._ensure_model():
            return None
        try:
            pairs = [(query, doc) for doc in docs]
            scores = self.model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as e:
            logger.exception(f"Reranking failed: {e}")
            return None


def rerank(query: str, docs: list[str], top_k: int = 2) -> list[str]:
    """
    Convenience function for simple reranking.
    Used directly in retriever without instantiating class.
    """
    if not docs:
        return []
    r = Reranker()
    scores = r.score_pairs(query, docs)
    if scores is None:
        return docs[:top_k]
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]