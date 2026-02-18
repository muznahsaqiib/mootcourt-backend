"""Reranker using a cross-encoder from sentence-transformers.

This module lazy-loads the CrossEncoder to avoid heavy imports at startup.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker wrapper.

    Usage:
        r = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = r.score_pairs(query, docs)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception:
                logger.warning("CrossEncoder not available (install sentence-transformers>=2.2.2). Reranking disabled.")
                return False

            try:
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self.model = CrossEncoder(self.model_name)
                return True
            except Exception as e:
                logger.exception("Failed to load CrossEncoder model; reranking disabled")
                self.model = None
                return False
        return True

    def available(self) -> bool:
        return self._ensure_model()

    def score_pairs(self, query: str, docs: List[str]) -> Optional[List[float]]:
        """Return a score per doc for (query, doc) using the cross-encoder.

        Returns None if the model cannot be loaded.
        """
        if not self._ensure_model():
            return None

        # prepare pairs
        pairs = [(query, d) for d in docs]
        try:
            scores = self.model.predict(pairs)
            return [float(s) for s in scores]
        except Exception:
            logger.exception("Cross-encoder scoring failed")
            return None
