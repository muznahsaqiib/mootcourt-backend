"""
Lightweight RAG Evaluation Script
File: eval_rag/evaluator/rag_evaluation.py
Run: python -m eval_rag.evaluator.rag_evaluation
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from eval_rag.embeddings.embedder import embed_query, embed
from eval_rag.db.chroma_client import legal_col


# ==============================
# TEST CASES
# Customize these to match your actual legal data
# ==============================
TEST_CASES = [
    {
        "query": "fundamental rights violation under Article 21",
        "expected_keywords": ["Article 21", "life", "liberty", "fundamental right"],
        "case_type": "constitutional"
    },
    {
        "query": "writ of mandamus against public authority",
        "expected_keywords": ["mandamus", "writ", "public duty", "court"],
        "case_type": "constitutional"
    },
    {
        "query": "jurisdiction of high court under Article 226",
        "expected_keywords": ["Article 226", "High Court", "jurisdiction", "writ"],
        "case_type": "constitutional"
    },
    {
        "query": "FIR registration under CrPC Section 154",
        "expected_keywords": ["FIR", "Section 154", "police", "cognizable"],
        "case_type": "criminal"
    },
    {
        "query": "bail application under Section 497 CrPC",
        "expected_keywords": ["bail", "Section 497", "accused", "court"],
        "case_type": "criminal"
    },
]


# ==============================
# METRIC 1: Context Relevance
# Cosine similarity between query and retrieved chunks
# Score: 0.0 – 1.0
# ==============================
def context_relevance(query: str, retrieved_docs: list[str]) -> float:
    if not retrieved_docs:
        return 0.0
    q_emb = np.array(embed_query(query)).reshape(1, -1)
    scores = []
    for doc in retrieved_docs:
        d_emb = np.array(embed(doc)).reshape(1, -1)
        score = cosine_similarity(q_emb, d_emb)[0][0]
        scores.append(score)
    return round(float(np.mean(scores)), 4)


# ==============================
# METRIC 2: Keyword Hit Rate
# Does retrieved context contain expected legal keywords?
# Score: 0.0 – 1.0
# ==============================
def hit_rate(retrieved_docs: list[str], expected_keywords: list[str]) -> float:
    if not retrieved_docs or not expected_keywords:
        return 0.0
    combined = " ".join(retrieved_docs).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return round(hits / len(expected_keywords), 4)


# ==============================
# METRIC 3: Retrieval Coverage
# How many chunks returned vs requested?
# Score: 0.0 – 1.0
# ==============================
def retrieval_coverage(retrieved_docs: list[str], n_requested: int) -> float:
    return round(len(retrieved_docs) / n_requested, 4) if n_requested > 0 else 0.0


# ==============================
# RUN EVALUATION
# ==============================
def run_evaluation():
    print("\n" + "=" * 60)
    print("         RAG EVALUATION REPORT")
    print("=" * 60)

    all_relevance, all_hitrate, all_coverage = [], [], []

    for i, tc in enumerate(TEST_CASES):
        print(f"\n📋 Test Case {i+1}: {tc['query'][:55]}...")

        # Retrieve from ChromaDB
        try:
            results = legal_col.query(
                query_embeddings=[embed_query(tc["query"])],
                n_results=3
            )
            docs = results["documents"][0] if results.get("documents") else []
        except Exception as e:
            print(f"  ⚠ Retrieval failed: {e}")
            docs = []

        # Compute metrics
        relevance = context_relevance(tc["query"], docs)
        hr        = hit_rate(docs, tc["expected_keywords"])
        coverage  = retrieval_coverage(docs, 3)

        all_relevance.append(relevance)
        all_hitrate.append(hr)
        all_coverage.append(coverage)

        print(f"  ✅ Context Relevance : {relevance:.2%}")
        print(f"  ✅ Keyword Hit Rate  : {hr:.2%}  {tc['expected_keywords']}")
        print(f"  ✅ Retrieval Coverage: {coverage:.2%} ({len(docs)}/3 chunks returned)")
        if docs:
            print(f"  📄 Top chunk preview : {docs[0][:150]}...")

    # Summary
    print("\n" + "=" * 60)
    print("  OVERALL SCORES")
    print("=" * 60)
    avg_relevance = np.mean(all_relevance)
    avg_hitrate   = np.mean(all_hitrate)
    avg_coverage  = np.mean(all_coverage)

    print(f"  Avg Context Relevance : {avg_relevance:.2%}")
    print(f"  Avg Keyword Hit Rate  : {avg_hitrate:.2%}")
    print(f"  Avg Retrieval Coverage: {avg_coverage:.2%}")
    print("=" * 60)

    overall = np.mean([avg_relevance, avg_hitrate, avg_coverage])
    grade = (
        "🟢 Excellent" if overall > 0.75 else
        "🟡 Good"      if overall > 0.50 else
        "🔴 Needs Work"
    )
    print(f"\n  Overall RAG Grade: {grade} ({overall:.2%})\n")


if __name__ == "__main__":
    run_evaluation()