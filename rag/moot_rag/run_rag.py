# run_opponent_rag.py
import logging
import re
import threading
from typing import Dict, Tuple, Optional

from rag.moot_rag.retrieval.hybrid_retriever import HybridRetriever
from rag.moot_rag.retrieval.rerank_utils import rerank_if_available
from rag.moot_rag.embeddings.embedder import embed_fn
from rag.moot_rag.llm.groq_rebuttal import generate_rebuttal
from rag.moot_rag.database_ch.chroma_client import collection

logger = logging.getLogger(__name__)

_retriever_cache: Dict[Tuple[str, Optional[str], bool, float, int], HybridRetriever] = {}
_retriever_cache_lock = threading.Lock()


def _get_retriever(
    *,
    case_key: str,
    case_type: Optional[str],
    include_legal_docs: bool,
    alpha: float,
    top_k: int,
) -> HybridRetriever:
    cache_key = (case_key, case_type, include_legal_docs, float(alpha), int(top_k))
    with _retriever_cache_lock:
        if cache_key in _retriever_cache:
            return _retriever_cache[cache_key]
        if len(_retriever_cache) > 32:
            _retriever_cache.clear()
        r = HybridRetriever(
            collection=collection,
            embed_fn=embed_fn,
            case_key=case_key,
            case_type=case_type,
            include_legal_docs=include_legal_docs,
            alpha=alpha,
            top_k=top_k,
        )
        _retriever_cache[cache_key] = r
        return r


def _format_sources(docs: list) -> str:
    """Format retrieved docs with source IDs for citation."""
    lines = []
    for d in docs:
        meta = d.get("meta") or {}
        parts = [meta.get("source_type", ""), meta.get("case_key", ""), meta.get("case_title", "")]
        header = " | ".join(p for p in parts if p) or "source"
        lines.append(f"SOURCE {d.get('id', '')} ({header}):\n{d.get('doc', '')}")
    return "\n\n".join(lines)


def _is_meaningful_input(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False

    greeting_re = r'^(hi|hello|hey|yo|sup|good\s+morning|good\s+afternoon|good\s+evening)[\.\!\?]*$'
    if re.match(greeting_re, text.strip(), flags=re.I):
        return False

    return True


def run_opponent_rag(case_key: str, argument: str, history: list, case_type: str = None):
    """
    Run RAG for respondent/opponent with full context:
    - Retrieves by case_key, optional case_type, and legal docs
    - Performs hybrid dense + sparse retrieval
    - Optional cross-encoder reranking
    """

    if not _is_meaningful_input(argument):
        return {
            "response": "Please provide a substantive legal argument or question.",
            "sources": []
        }

    # -----------------------------
    # Build conversation history
    # -----------------------------
    history_text = ""
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user").upper()
            text = msg.get("content", msg.get("text", ""))
        else:
            role = getattr(msg, "role", "user").upper()
            text = getattr(msg, "text", "")

        history_text += f"{role}: {text}\n"

    # -----------------------------
    # Retrieval (cached per case) + rerank
    # -----------------------------
    retriever = _get_retriever(
        case_key=case_key,
        case_type=case_type,
        include_legal_docs=True,
        alpha=0.6,
        top_k=20,
    )
    retrieved_docs, _ = retriever.retrieve(argument)
    logger.info("Retrieved %d docs for argument=%.50s", len(retrieved_docs), argument)

    retrieved_docs = rerank_if_available(query=argument, docs=retrieved_docs, final_k=8)
    logger.info("Reranked to %d docs", len(retrieved_docs))

    retrieved_text = _format_sources(retrieved_docs)
    final_context = (
        f"CASE_KEY: {case_key}\nCASE_TYPE: {case_type}\n\n"
        f"HISTORY:\n{history_text}\n\n"
        f"RETRIEVED MATERIAL (CITE ONLY THESE SOURCES BY ID):\n{retrieved_text}"
    )

    # -----------------------------
    # LLM Generation
    # -----------------------------
    rebuttal = generate_rebuttal(
        argument=argument,
        context=final_context
    )

    return {
        "response": rebuttal,
        "sources": [d.get("meta", {}) for d in retrieved_docs]
    }
