# run_opponent_rag.py
import logging
import re

from rag.moot_rag.retrieval.hybrid_retriever import HybridRetriever
from rag.moot_rag.retrieval.rerank_utils import rerank_if_available
from rag.moot_rag.embeddings.embedder import embed_fn
from rag.moot_rag.llm.groq_rebuttal import generate_rebuttal
from rag.moot_rag.database_ch.chroma_client import collection

logger = logging.getLogger(__name__)


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
    # Retrieval using updated HybridRetriever
    # -----------------------------
    retriever = HybridRetriever(
        collection=collection,
        embed_fn=embed_fn,
        case_key=case_key,
        case_type=case_type,        # pass case_type if available
        include_legal_docs=True,    # always include legal docs for stronger context
        top_k=20                    # high recall
    )

    retrieved_docs, _ = retriever.retrieve(argument)
    print(f"📝 Retrieved {len(retrieved_docs)} docs for argument: {argument[:50]}")

    # -----------------------------
    # Reranking (cross-encoder)
    # -----------------------------
    retrieved_docs = rerank_if_available(
        query=argument,
        docs=retrieved_docs,
        final_k=8
    )
    print(f"🔝 Reranked to {len(retrieved_docs)} top docs")

    retrieved_text = "\n\n".join(d["doc"] for d in retrieved_docs)

    final_context = (
        f"CASE_KEY: {case_key}\nCASE_TYPE: {case_type}\n\n"
        f"HISTORY:\n{history_text}\n\n"
        f"RETRIEVED MATERIAL:\n{retrieved_text}"
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
