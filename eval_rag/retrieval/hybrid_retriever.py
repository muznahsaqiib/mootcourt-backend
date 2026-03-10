# rag/moot_rag/retrieval/hybrid_retriever.py
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple, Optional
import threading

logger = logging.getLogger(__name__)


def _where_case(case_key: str, case_type: str = None) -> Dict[str, Any]:
    if case_type:
        return {"$and": [{"case_key": {"$eq": case_key}}, {"case_type": {"$eq": case_type}}]}
    return {"case_key": {"$eq": case_key}}


def _where_source_type(source_type: str) -> Dict[str, Any]:
    return {"source_type": {"$eq": source_type}}


# ===============================
# Query Expansion (shared with evaluator)
# ===============================
def _expand_query(query: str, case_type: str = "") -> list[str]:
    expansions = [query]
    query_lower = query.lower()

    if "article 21" in query_lower or "life" in query_lower:
        expansions.append("right to life personal liberty fundamental rights protection")
    if "article 226" in query_lower or "writ" in query_lower:
        expansions.append("high court writ jurisdiction mandamus certiorari prohibition")
    if "fundamental right" in query_lower:
        expansions.append("constitutional guarantee enforceable rights citizen state")
    if "fir" in query_lower or "section 154" in query_lower:
        expansions.append("first information report police registration cognizable offence")
    if "bail" in query_lower:
        expansions.append("bail application accused detention release surety")
    if "crpc" in query_lower or "criminal procedure" in query_lower:
        expansions.append("criminal procedure code Pakistan arrest investigation trial")

    if case_type:
        ct = case_type.lower()
        if ct == "constitutional":
            expansions.append("constitutional law fundamental rights Pakistan judiciary")
        elif ct == "criminal":
            expansions.append("criminal law Pakistan penal code procedure offence")
        elif ct == "civil":
            expansions.append("civil procedure code Pakistan suit decree appeal")

    return expansions[:3]


class HybridRetriever:
    """
    Hybrid Retriever:
    - Dense retrieval (Legal-BERT via Chroma)
    - Sparse retrieval (BM25)
    - Query expansion (averaged embeddings)
    - Dynamic alpha per query type
    - BM25 built ONCE at init
    - Deduplication at load time
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

    def _load_case_docs(self):
        logger.info(f"[HybridRetriever] Loading docs case_key={self.case_key} case_type={self.case_type}")

        where_case = _where_case(self.case_key, self.case_type)
        res = self.collection.get(where=where_case)

        documents = res.get("documents", []) if res else []
        metadatas = res.get("metadatas", []) if res else []
        ids       = res.get("ids", []) if res else []

        if self.include_legal_docs:
            legal_res = self.collection.get(where=_where_source_type("law"))
            if legal_res and legal_res.get("documents"):
                documents += legal_res["documents"]
                metadatas += legal_res.get("metadatas", [])
                ids       += legal_res.get("ids", [])

        if not documents:
            logger.warning(f"No documents found for case_key={self.case_key}")
            return

        # Deduplicate at load time
        seen_content = set()
        clean_docs, clean_metas, clean_ids = [], [], []
        for i, doc in enumerate(documents):
            content = doc.strip() if isinstance(doc, str) else ""
            if content and content not in seen_content:
                seen_content.add(content)
                clean_docs.append(content)
                clean_metas.append(metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {})
                clean_ids.append(str(ids[i]) if i < len(ids) else f"doc_{i}")

        self.doc_texts    = clean_docs
        self.doc_metadatas = clean_metas
        self.doc_ids      = clean_ids

        # Build BM25 once
        tokenized_docs = [doc.lower().split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"[HybridRetriever] Loaded {len(self.doc_texts)} unique chunks")

    def _get_alpha(self, query: str) -> float:
        query_lower = query.lower()
        criminal_signals      = ["section", "crpc", "ppc", "fir", "bail", "arrest", "accused", "offence"]
        constitutional_signals = ["fundamental right", "article", "constitution", "writ", "mandamus", "certiorari"]
        criminal_hits      = sum(1 for s in criminal_signals if s in query_lower)
        constitutional_hits = sum(1 for s in constitutional_signals if s in query_lower)
        if criminal_hits > constitutional_hits:
            return 0.4
        elif constitutional_hits > criminal_hits:
            return 0.7
        return self.alpha

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], float]:
        if not self.doc_texts:
            return [], 0.0

        logger.info(f"[HybridRetriever] Query: {query[:100]}")
        alpha = self._get_alpha(query)

        # ✅ Query expansion — average embeddings of all expanded queries
        expanded_queries = _expand_query(query, self.case_type or "")
        all_embeddings = [
            self.embed_fn([f"legal query: {q}"])[0]
            for q in expanded_queries
        ]
        query_embedding = np.mean(all_embeddings, axis=0).tolist()

        # ----------------------
        # Dense retrieval
        # ----------------------
        n_dense    = max(self.top_k * 3, 30)
        case_where = _where_case(self.case_key, self.case_type)
        dense_scores = {}

        try:
            case_dense = self.collection.query(
                query_embeddings=[query_embedding],
                where=case_where,
                n_results=n_dense,
                include=["distances"],
            )
            if case_dense and case_dense.get("ids") and case_dense.get("distances"):
                for i, doc_id in enumerate(case_dense["ids"][0]):
                    sim = 1 - float(case_dense["distances"][0][i])
                    dense_scores[str(doc_id)] = max(dense_scores.get(str(doc_id), 0), sim)
        except Exception as e:
            logger.warning(f"Dense case retrieval failed: {e}")

        if self.include_legal_docs:
            try:
                law_dense = self.collection.query(
                    query_embeddings=[query_embedding],
                    where=_where_source_type("law"),
                    n_results=max(self.top_k, 15),
                    include=["distances"],
                )
                if law_dense and law_dense.get("ids") and law_dense.get("distances"):
                    for i, doc_id in enumerate(law_dense["ids"][0]):
                        sim = 1 - float(law_dense["distances"][0][i])
                        dense_scores[str(doc_id)] = max(dense_scores.get(str(doc_id), 0), sim)
            except Exception as e:
                logger.warning(f"Dense law retrieval failed: {e}")

        # Normalize dense scores
        if dense_scores:
            values = np.array(list(dense_scores.values()))
            min_v, max_v = values.min(), values.max()
            for k in dense_scores:
                dense_scores[k] = (dense_scores[k] - min_v) / (max_v - min_v + 1e-6)

        # ----------------------
        # Sparse retrieval (BM25)
        # ----------------------
        sparse_scores = {}
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            max_score   = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            top_indices = np.argsort(bm25_scores)[-self.top_k:][::-1]
            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    sparse_scores[self.doc_ids[idx]] = bm25_scores[idx] / max_score

        # ----------------------
        # Merge scores
        # ----------------------
        merged_results = []
        all_ids = set(list(dense_scores.keys()) + list(sparse_scores.keys()))

        for doc_id in all_ids:
            if doc_id not in self.doc_ids:
                continue
            idx      = self.doc_ids.index(doc_id)
            doc_text = self.doc_texts[idx]
            metadata = self.doc_metadatas[idx]

            d_score     = dense_scores.get(doc_id, 0)
            s_score     = sparse_scores.get(doc_id, 0)
            final_score = alpha * d_score + (1 - alpha) * s_score

            # Score boosting
            doc_lower = doc_text.lower()
            if metadata.get("statute") or metadata.get("case_ref"):
                final_score += 0.05
            if any(k in doc_lower for k in ["article", "section", "constitution", "act", "writ"]):
                final_score += 0.03
            if self.case_type and metadata.get("case_type") == self.case_type:
                final_score += 0.05

            merged_results.append({
                "id":    doc_id,
                "doc":   doc_text,
                "meta":  metadata,
                "score": final_score
            })

        merged_results.sort(key=lambda x: x["score"], reverse=True)
        ranked    = merged_results[:self.top_k]
        top_score = ranked[0]["score"] if ranked else 0.0

        logger.info(f"[HybridRetriever] top_score={top_score:.4f} alpha={alpha}")
        return ranked, top_score


# ===============================
# Thread-safe retriever cache
# ===============================
_retriever_cache: Dict[Tuple, HybridRetriever] = {}
_retriever_cache_lock = threading.Lock()


def get_retriever(
    collection,
    embed_fn,
    case_key: str,
    case_type: str = None,
    include_legal_docs: bool = True,
    alpha: float = 0.6,
    top_k: int = 15
) -> HybridRetriever:
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
            alpha=alpha,
            top_k=top_k
        )
        _retriever_cache[cache_key] = r
        return r