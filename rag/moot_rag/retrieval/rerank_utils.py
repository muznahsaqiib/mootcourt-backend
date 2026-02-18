from typing import List, Dict, Any
from rag.moot_rag.retrieval.reranker import Reranker
import logging

logger = logging.getLogger(__name__)

_reranker = Reranker()

def rerank_if_available(
    query: str,
    docs: List[Dict[str, Any]],
    final_k: int = 8
) -> List[Dict[str, Any]]:
    """
    Apply cross-encoder reranking if available.
    Expects docs in format:
      { "doc": str, "meta": dict, "score": float }
    """

    if not docs:
        return docs

    if not _reranker.available():
        logger.info("Reranker unavailable, using hybrid scores only")
        return docs[:final_k]

    texts = [d["doc"] for d in docs]

    scores = _reranker.score_pairs(query, texts)
    if scores is None:
        return docs[:final_k]

    # Attach rerank scores
    for d, s in zip(docs, scores):
        d["rerank_score"] = s

    # Sort purely by cross-encoder
    docs.sort(key=lambda x: x["rerank_score"], reverse=True)

    return docs[:final_k]
