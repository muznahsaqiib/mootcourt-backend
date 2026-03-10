# run_opponent_rag.py
import logging
import re
import threading
from typing import Dict, Tuple, Optional

from rag.moot_rag.retrieval.hybrid_retriever import HybridRetriever
from rag.moot_rag.retrieval.rerank_utils import rerank_if_available
from rag.moot_rag.embeddings.embedder import embed_fn
from rag.moot_rag.llm.groq_rebuttal import generate_rebuttal, generate_judge_reply
from rag.moot_rag.database_ch.chroma_client import collection

logger = logging.getLogger(__name__)

_retriever_cache: Dict[Tuple[str, Optional[str], bool, int], HybridRetriever] = {}
_retriever_cache_lock = threading.Lock()


def _get_retriever(*, case_key, case_type, include_legal_docs, top_k) -> HybridRetriever:
    cache_key = (case_key, case_type, include_legal_docs, int(top_k))
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
            alpha=0.6,
            top_k=top_k,
        )
        _retriever_cache[cache_key] = r
        return r


def _is_meaningful_input(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return False
    greeting_re = r'^(hi|hello|hey|yo|sup|good\s+morning|good\s+afternoon|good\s+evening)[\.\!\?]*$'
    if re.match(greeting_re, text.strip(), flags=re.I):
        return False
    return True


def _build_history_text(history: list) -> str:
    lines = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user").upper()
            text = msg.get("content", msg.get("text", ""))
        else:
            role = getattr(msg, "role", "user").upper()
            text = getattr(msg, "text", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _retrieve_and_rerank(case_key, case_type, query, top_k=15, final_k=5):
    """Shared retrieval + rerank logic."""
    try:
        retriever = _get_retriever(
            case_key=case_key,
            case_type=case_type,
            include_legal_docs=True,
            top_k=top_k,
        )
        docs, top_score = retriever.retrieve(query)
        logger.info("Retrieved %d docs | top_score=%.4f", len(docs), top_score)
    except Exception as e:
        logger.exception("Retrieval failed")
        docs = []

    try:
        docs = rerank_if_available(query=query, docs=docs, final_k=final_k)
        logger.info("Reranked to %d docs", len(docs))
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        docs = docs[:final_k]

    return docs


# ===============================
# MAIN RESPONDENT ARGUMENT
# ===============================
def run_opponent_rag(
    case_key: str,
    argument: str,
    history: list,
    case_type: str = None,
    case_summary: str = "",    # ✅ new
    case_title: str = ""       # ✅ new
) -> dict:

    if not _is_meaningful_input(argument):
        return {
            "response": "Please provide a substantive legal argument or question.",
            "sources": []
        }

    history_text = _build_history_text(history)
    retrieved_docs = _retrieve_and_rerank(case_key, case_type, argument)

    # ✅ Build preamble with full case context
    preamble_parts = [
        f"CASE: {case_title}" if case_title else "",
        f"CASE_KEY: {case_key}",
        f"CASE_TYPE: {case_type or 'general'}",
        f"CASE SUMMARY:\n{case_summary}" if case_summary else "",
        f"HEARING HISTORY:\n{history_text}" if history_text.strip() else ""
    ]
    preamble = "\n\n".join(p for p in preamble_parts if p)

    rebuttal = generate_rebuttal(
        argument=argument,
        context=retrieved_docs,
        party="respondent",
        preamble=preamble
    )

    return {
        "response": rebuttal,
        "sources": [d.get("meta", {}) for d in retrieved_docs]
    }


# ===============================
# JUDGE REPLY — short, focused
# ✅ New function — does NOT use full argument prompt
# ===============================
def run_judge_reply(
    case_key: str,
    judge_question: str,
    history: list,
    case_type: str = None,
    case_summary: str = ""
) -> dict:

    # Retrieve small focused context for judge question only
    retrieved_docs = _retrieve_and_rerank(
        case_key, case_type, judge_question,
        top_k=10, final_k=3    # smaller — judge reply needs less context
    )

    reply = generate_judge_reply(
        question=judge_question,
        context=retrieved_docs,
        case_summary=case_summary,
        party="respondent"
    )

    return {
        "response": reply,
        "sources": [d.get("meta", {}) for d in retrieved_docs]
    }