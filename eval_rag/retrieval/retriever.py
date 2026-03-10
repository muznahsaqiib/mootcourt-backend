"""
eval_rag/retrieval/retriever.py
Upgrades: Query expansion + case summary injection
"""
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from eval_rag.embeddings.embedder import embed, embed_query
from eval_rag.db.chroma_client import legal_col, judge_col, evaluated_col
from eval_rag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)
_reranker = Reranker()


# ===============================
# Query Expansion
# ===============================
def _expand_query(query: str, case_type: str) -> list[str]:
    """
    Generate alternative query phrasings to improve retrieval coverage.
    Returns original + up to 2 expanded versions.
    """
    expansions = [query]
    query_lower = query.lower()

    # Constitutional expansions
    if "article 21" in query_lower or "life" in query_lower:
        expansions.append("right to life personal liberty fundamental rights protection")
    if "article 226" in query_lower or "writ" in query_lower:
        expansions.append("high court writ jurisdiction mandamus certiorari prohibition")
    if "fundamental right" in query_lower:
        expansions.append("constitutional guarantee enforceable rights citizen state")

    # Criminal expansions
    if "fir" in query_lower or "section 154" in query_lower:
        expansions.append("first information report police registration cognizable offence")
    if "bail" in query_lower:
        expansions.append("bail application accused detention release surety")
    if "crpc" in query_lower or "criminal procedure" in query_lower:
        expansions.append("criminal procedure code Pakistan arrest investigation trial")

    # Case type expansion
    if case_type:
        ct = case_type.lower()
        if ct == "constitutional":
            expansions.append(f"constitutional law fundamental rights Pakistan judiciary")
        elif ct == "criminal":
            expansions.append(f"criminal law Pakistan penal code procedure offence")
        elif ct == "civil":
            expansions.append(f"civil procedure code Pakistan suit decree appeal")

    return expansions[:3]  # original + max 2 expansions


# ===============================
# Retrieval with Query Expansion
# ===============================
def _retrieve_with_expansion(
    collection,
    query: str,
    case_type: str,
    n_results: int = 3
) -> list[str]:
    """
    Retrieve using expanded queries — merges results from all variants.
    Deduplicates by content.
    """
    expanded_queries = _expand_query(query, case_type)
    seen = set()
    all_docs = []

    for q in expanded_queries:
        try:
            hits = collection.query(
                query_embeddings=[embed_query(q)],
                n_results=n_results
            )
            if hits and hits.get("documents"):
                for doc_list in hits["documents"]:
                    for doc in doc_list:
                        if isinstance(doc, str) and doc.strip() and doc not in seen:
                            seen.add(doc)
                            all_docs.append(doc)
        except Exception as e:
            logger.warning(f"Expansion query failed for '{q[:50]}': {e}")

    return all_docs


# ===============================
# Relevance Filter
# ===============================
def _is_legal_chunk_relevant(text: str) -> bool:
    keywords = [
        "Article", "Constitution", "jurisdiction",
        "fundamental right", "writ", "court",
        "duty", "authority", "statute", "provision",
        "Section", "Act", "tribunal", "petition"
    ]
    matches = sum(1 for k in keywords if k.lower() in text.lower())
    return matches >= 2


# ===============================
# Main retrieve_context
# ===============================
def retrieve_context(
    main_argument: str,
    judge_question: str,
    case_type: str,
    case_summary: str = "",       # ✅ new
    n_eval_examples: int = 2
) -> str:

    law_query_text = f"{case_type}: {main_argument[:300]}"

    # ===============================
    # 1️⃣ Legal context — Hybrid + Query Expansion + Rerank
    # ===============================
    legal_docs = []
    try:
        # Expanded candidate pool
        expanded_candidates = _retrieve_with_expansion(
            legal_col, law_query_text, case_type, n_results=3
        )

        # Filter for relevance
        candidates = [d for d in expanded_candidates if _is_legal_chunk_relevant(d)]

        # Rerank expanded pool
        if _reranker.available() and candidates:
            scores = _reranker.score_pairs(law_query_text, candidates)
            if scores:
                ranked = sorted(
                    zip(scores, candidates),
                    key=lambda x: x[0],
                    reverse=True
                )
                legal_docs = [doc for _, doc in ranked[:2]]
        else:
            legal_docs = candidates[:2]

    except Exception as e:
        logger.warning(f"Legal retrieval failed: {e}")

    legal_text = "\n\n".join(legal_docs)
    if not legal_text:
        legal_text = (
            "NO LEGAL PROVISIONS RETRIEVED. "
            "Do NOT cite any statutes, articles, or case law. "
            "Evaluate based solely on argument quality and structure."
        )

    # ===============================
    # 2️⃣ Judge context — dense only
    # ===============================
    judge_text = ""
    if judge_question:
        try:
            judge_emb = embed_query(judge_question)
            judge_hits = judge_col.query(
                query_embeddings=[judge_emb],
                n_results=2,
                where={"case_type": case_type}
            )
            if judge_hits and judge_hits.get("documents"):
                docs = judge_hits["documents"][0]
                judge_text = docs[0] if docs else ""
        except Exception as e:
            logger.warning(f"Judge retrieval failed: {e}")

    # ===============================
    # 3️⃣ Few-shot evaluated arguments
    # ===============================
    few_shot_args = []
    try:
        arg_emb = embed_query(main_argument)
        eval_hits = evaluated_col.query(
            query_embeddings=[arg_emb],
            n_results=n_eval_examples,
            where={
                "$and": [
                    {"case_type": case_type},
                    {"round_type": "oral_arguments"}
                ]
            }
        )
        if eval_hits and eval_hits.get("documents"):
            for doc_list in eval_hits["documents"]:
                for doc in doc_list:
                    if isinstance(doc, str) and doc.strip():
                        few_shot_args.append(doc)
    except Exception as e:
        logger.warning(f"Eval retrieval failed: {e}")

    few_shot_text = "\n\n".join(
        [f"EXAMPLE ARGUMENT (style reference only):\n{t}" for t in few_shot_args]
    ) if few_shot_args else "No prior example arguments retrieved."

    # ===============================
    # 4️⃣ Assemble structured context
    # ✅ Case summary injected at top
    # ===============================
    return f"""
[CASE SUMMARY — factual background]
{case_summary if case_summary else "No case summary available."}

[JUDGE CONTEXT — scope & focus]
{judge_text if judge_text else "No prior judge guidance available."}

[APPLICABLE LEGAL PROVISIONS — authoritative]
{legal_text}

[EXAMPLE ARGUMENT STYLE — reference only, do not copy scores]
{few_shot_text}
""".strip()