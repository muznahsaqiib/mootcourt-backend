from ..embeddings.embedder import embed
from ..db.chroma_client import legal_col, judge_col, evaluated_col

def retrieve_context(
    main_argument: str,
    judge_question: str,
    case_type: str,
    n_eval_examples: int = 2  # top 2 arguments as few-shot examples
):
    """
    High-precision retrieval for evaluator RAG with safe few-shot examples.
    """

    # ===============================
    # 1️⃣ Intent-specific embeddings
    # ===============================
    argument_emb = embed(main_argument)
    judge_emb = embed(judge_question) if judge_question else None

    law_query_text = f"""
    {case_type} constitutional principles
    jurisdiction fundamental rights
    relevant Articles and legal standards
    """
    law_emb = embed(law_query_text)

    # ===============================
    # 2️⃣ Retrieve JUDGE CONTEXT (k=1)
    # ===============================
    judge_text = ""
    if judge_emb:
        judge_hits = judge_col.query(
            query_embeddings=[judge_emb],
            n_results=1,
            where={"case_type": case_type}
        )
        if judge_hits and judge_hits.get("documents"):
            judge_text = judge_hits["documents"][0]

    # ===============================
    # 3️⃣ Retrieve LEGAL CONTEXT (k=2)
    # ===============================
    legal_hits = legal_col.query(
        query_embeddings=[law_emb],
        n_results=2
    )
    legal_docs = []
    if legal_hits and legal_hits.get("documents"):
        for doc_list in legal_hits["documents"]:
            for doc in doc_list:
                if isinstance(doc, str) and _is_legal_chunk_relevant(doc):
                    legal_docs.append(doc)
    legal_text = "\n\n".join(legal_docs)

    # ===============================
    # 4️⃣ Retrieve few-shot evaluated arguments (safe)
    # ===============================
    eval_hits = evaluated_col.query(
    query_embeddings=[argument_emb],
    n_results=n_eval_examples,
    where={
        "$and": [
            {"case_type": case_type},
            {"round_type": "oral_arguments"}
        ]
    }
)
    # Only take the raw argument text, NO scores, NO justification
    few_shot_args = []
    if eval_hits and eval_hits.get("documents"):
        for doc_list in eval_hits["documents"]:
            if not isinstance(doc_list, list):
                continue
            for doc in doc_list:
                if isinstance(doc, str) and doc.strip():
                    few_shot_args.append(doc)

    few_shot_text = "\n\n".join(
        [f"EXAMPLE ARGUMENT (style reference):\n{text}" for text in few_shot_args]
    )

    # ===============================
    # 5️⃣ Structured Context Assembly
    # ===============================
    structured_context = f"""
[JUDGE CONTEXT — scope & focus]
{judge_text if judge_text else "No prior judge guidance available."}

[APPLICABLE LEGAL PROVISIONS — authoritative]
{legal_text if legal_text else "No directly applicable legal provision retrieved."}

[FOLLOW ARGUMENT STYLE — few-shot examples]
{few_shot_text if few_shot_text else "No prior example arguments retrieved."}
""".strip()

    return structured_context


# ===============================
# Utility: Law relevance filter
# ===============================
def _is_legal_chunk_relevant(text: str) -> bool:
    keywords = [
        "Article",
        "Constitution",
        "jurisdiction",
        "fundamental right",
        "writ",
        "court",
        "duty",
        "authority"
    ]
    text_lower = text.lower()
    return any(k.lower() in text_lower for k in keywords)
