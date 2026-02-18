import logging
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid Retriever with full context:
    - Dense retrieval (Legal-BERT embeddings via Chroma)
    - Sparse retrieval (BM25)
    - Can filter by case_key, case_type, and include legal docs
    """
    def __init__(
        self,
        collection,
        embed_fn,
        case_key: str,
        case_type: str = None,
        include_legal_docs: bool = True,
        alpha: float = 0.6,
        top_k: int = 15
    ):
        self.collection = collection
        self.embed_fn = embed_fn
        self.case_key = case_key
        self.case_type = case_type
        self.include_legal_docs = include_legal_docs
        self.alpha = alpha
        self.top_k = top_k

        self.doc_texts: List[str] = []
        self.doc_metadatas: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []
        self.bm25 = None

        self._load_case_docs()

    # -----------------------------
    # Load all relevant chunks
    # -----------------------------
    def _load_case_docs(self):
        logger.info(f"[HybridRetriever] Loading docs for case_key={self.case_key} case_type={self.case_type}")

        filters = {"case_key": self.case_key}
        if self.case_type:
            filters["case_type"] = self.case_type

        res = self.collection.get(
            where=filters,
            include=["documents", "metadatas"]
        )

        documents = res.get("documents", []) if res else []
        metadatas = res.get("metadatas", []) if res else []

        # Include legal docs
        if self.include_legal_docs:
            legal_res = self.collection.get(
                where={"source_type": "law"},
                include=["documents", "metadatas"]
            )

            if legal_res and legal_res.get("documents"):
                documents += legal_res["documents"]
                metadatas += legal_res["metadatas"]

        if not documents:
            logger.warning(f"No documents found for case_key={self.case_key}")
            return

        # Clean texts
        self.doc_texts = [d.strip() for d in documents]
        self.doc_metadatas = metadatas

        # 🔥 Generate stable local IDs (fixes index mismatch permanently)
        self.doc_ids = [f"doc_{i}" for i in range(len(self.doc_texts))]

        # Build BM25
        tokenized_docs = [doc.lower().split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)

        logger.info(f"[HybridRetriever] Loaded {len(self.doc_texts)} chunks (BM25 built)")


    # -----------------------------
    # Retrieval method
    # -----------------------------
    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], float]:
        if not self.doc_texts:
            return [], 0.0

        logger.info(f"[HybridRetriever] Query: {query[:100]}")

        # ----------------------
        # Dense retrieval
        # ----------------------
        query_embedding = self.embed_fn([query])[0]

        dense_res = self.collection.query(
        query_embeddings=[query_embedding],
        where={"case_key": self.case_key} if not self.case_type else {
            "case_key": self.case_key,
            "case_type": self.case_type
        },
        n_results=max(self.top_k * 3, 30),
        include=["documents", "metadatas", "distances"]  # ❌ remove "ids"
    )

        dense_scores = {}
        if dense_res and dense_res.get("ids"):
            for i, doc_id in enumerate(dense_res["ids"][0]):
                similarity = 1 - dense_res["distances"][0][i]
                dense_scores[doc_id] = similarity

        # normalize dense scores
        if dense_scores:
            values = np.array(list(dense_scores.values()))
            min_v, max_v = values.min(), values.max()
            for k in dense_scores:
                dense_scores[k] = (dense_scores[k] - min_v) / (max_v - min_v + 1e-6)

        # ----------------------
        # Sparse retrieval
        # ----------------------
        sparse_scores = {}
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            top_indices = np.argsort(bm25_scores)[-self.top_k:][::-1]

            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    sparse_scores[self.doc_ids[idx]] = bm25_scores[idx] / max_score

        # ----------------------
        # Merge dense + sparse
        # ----------------------
        merged_results = []
        for i, doc_id in enumerate(self.doc_ids):
            dense_score = dense_scores.get(doc_id, 0)
            sparse_score = sparse_scores.get(doc_id, 0)
            final_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score

            metadata = self.doc_metadatas[i]

            # Boost legal authority or case_type match
            if metadata.get("statute") or metadata.get("case_ref"):
                final_score += 0.05
            if self.case_type and metadata.get("case_type") == self.case_type:
                final_score += 0.05

            merged_results.append({
                "id": doc_id,
                "doc": self.doc_texts[i],
                "meta": metadata,
                "score": final_score
            })

        # sort and return top_k
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        ranked = merged_results[:self.top_k]
        top_score = ranked[0]["score"] if ranked else 0.0

        logger.info(f"[HybridRetriever] Top score: {top_score:.4f}")

        return ranked, top_score
