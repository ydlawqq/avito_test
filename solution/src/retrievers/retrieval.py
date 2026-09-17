"""Ретриверы для candidate generation: BM25, dense (faiss) и гибридный (RRF).

Все ретриверы возвращают одинаковый интерфейс — список RetrievedDoc,
поэтому их можно свободно комбинировать (см. HybridRetriever).

Используемые open-source библиотеки / модели:
  * rank_bm25 (BM25Okapi)        — лексический поиск
  * faiss (IndexIDMap + FlatIP)  — поиск по эмбеддингам
  * sentence-transformers        — би-энкодер (по умолчанию multilingual-e5)
  * pymorphy3                    — лемматизация (см. src/data/preprocessing.py)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

try:  # опциональные зависимости dense-ретривера
    import faiss

    _HAS_FAISS = True
except ImportError:  # pragma: no cover
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:  # pragma: no cover
    _HAS_ST = False


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    source: str  # "bm25" / "dense" / "hybrid"
    metadata: dict | None = None


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists via RRF.

    rrf_score(doc) = sum over rankings of 1 / (k + rank_in_ranking)

    Args:
        rankings: list of lists where each inner list is doc_ids ordered best-first
        k: smoothing constant, default 60 (standard)

    Returns:
        list of (doc_id, fused_score) sorted by score descending
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25Retriever:
    """BM25-индекс в памяти поверх предобработанных (лемматизированных) токенов.

    Корпус подаётся уже токенизированным: лемматизация 189k документов —
    дорогая операция, она выполняется один раз на этапе prepare_data.
    """

    def __init__(
        self,
        doc_ids: list[str],
        tokenized_docs: list[list[str]],
        texts: list[str] | None = None,
        metadatas: list[dict] | None = None,
    ):
        self.doc_ids = list(doc_ids)
        self.corpus = [
            {
                "id": d,
                "text": (texts[i] if texts else " ".join(tokenized_docs[i])),
                "metadata": (metadatas[i] if metadatas else None),
            }
            for i, d in enumerate(doc_ids)
        ]
        self.bm25 = BM25Okapi(tokenized_docs)

    @classmethod
    def from_documents(cls, corpus: list[dict]) -> "BM25Retriever":
        """Совместимость с исходной заготовкой: корпус из dict с ключами id/text."""
        return cls(
            doc_ids=[d["id"] for d in corpus],
            tokenized_docs=[d["text"].lower().split() for d in corpus],
        )

    # -- основной API -------------------------------------------------------

    def score_tokens(self, tokens: list[str]) -> np.ndarray:
        """Скор BM25 для всех документов корпуса по токенам запроса."""
        return self.bm25.get_scores(tokens)

    def top_indices(self, tokens: list[str], top_k: int = 50) -> list[int]:
        """Индексы top-k документов по BM25 (score > 0)."""
        scores = self.score_tokens(tokens)
        top = np.argsort(-scores)[:top_k]
        return [int(i) for i in top if scores[i] > 0]

    def search_tokens(
        self,
        tokens: list[str],
        top_k: int = 50,
    ) -> list[RetrievedDoc]:
        """Поиск по предобработанному (лемматизированному) запросу."""
        idx = self.top_indices(tokens, top_k)
        scores = self.bm25.get_scores(tokens)
        return [
            RetrievedDoc(
                doc_id=self.doc_ids[i],
                text=self.corpus[i]["text"],
                score=float(scores[i]),
                source="bm25",
                metadata=self.corpus[i]["metadata"],
            )
            for i in idx
        ]

    def search(self, query: str, top_k: int = 50) -> list[RetrievedDoc]:
        """Поиск по сырому тексту запроса (простая токенизация без лемматизации).

        В основном пайплайне используйте search_tokens — с лемматизацией метрика выше.
        """
        return self.search_tokens(query.lower().split(), top_k=top_k)



# ---------------------------------------------------------------------------
# Dense retrieval (faiss)
# ---------------------------------------------------------------------------


class DenseRetriever:
    """Dense retrieval: sentence-transformers би-энкодер + faiss-индекс.

    Индекс строится скриптом scripts/build_dense_index.py (GPU-машина) и
    хранится в artifacts/dense/. Чанкование — см. build_dense_index.py:
    на каждый item приходится 1..max_chunks чанков, при поиске чанки одного
    item схлопываются по максимальному скору.

    Модель по умолчанию — intfloat/multilingual-e5-large: сильный open-source
    би-энкодер для русского языка. E5 требует префиксов "query: "/"passage: ".
    """

    def __init__(
        self,
        index_path: str,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | None = None,
        query_prefix: str = "query: ",
    ):
        if not (_HAS_FAISS and _HAS_ST):
            raise ImportError(
                "Для DenseRetriever необходимы faiss и sentence-transformers: "
                "pip install faiss-cpu sentence-transformers"
            )
        import pathlib

        self.index = faiss.read_index(index_path)
        self.encoder = SentenceTransformer(model_name, device=device)
        self.query_prefix = query_prefix

        # Маппинг внутренний id чанка -> item_id лежит рядом с индексом
        mapping_path = pathlib.Path(index_path).with_name("chunk_item_ids.npy")
        self.chunk_item_ids: np.ndarray = np.load(mapping_path)  # type: ignore[assignment]
        self.item_ids: list[str] = self.chunk_item_ids.tolist()

    def embed_query(self, query: str) -> np.ndarray:
        # e5-модели ожидают префиксы "query: " / "passage: " для лучшего качества
        return self.encoder.encode(
            [f"{self.query_prefix}{query}"], normalize_embeddings=True
        )[0]

    def search(
        self,
        query: str,
        top_k: int = 50,
        oversample: int = 4,
    ) -> list[RetrievedDoc]:
        """Вернуть top_k уникальных item_id.

        oversample: чанков одного item несколько — достаём top_k*oversample
        чанков и схлопываем их по максимальному скору item'а.
        """
        vector = self.embed_query(query)
        scores, ids = self.index.search(
            vector.reshape(1, -1).astype("float32"), top_k * oversample
        )

        best: dict[str, float] = {}
        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id == -1:
                continue
            item_id = self.item_ids[chunk_id]
            best[item_id] = max(best.get(item_id, -np.inf), float(score))

        ranked = sorted(best.items(), key=lambda x: -x[1])[:top_k]
        return [
            RetrievedDoc(doc_id=i, text="", score=s, source="dense")
            for i, s in ranked
        ]


# ---------------------------------------------------------------------------
# Hybrid: BM25 + Dense через RRF
# ---------------------------------------------------------------------------


class HybridRetriever:
    """BM25 + dense, объединённые через Reciprocal Rank Fusion.

    RRF не требует нормализации сырых скоров (BM25-скор и косинусная близость
    несравнимы между собой), хорошо работает на практике и максимизирует
    полноту — именно то, что нужно для Recall@50.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever | None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        # Вес dense-ранжирования в RRF: 1/(k+r) умножается на dense_weight
        self.dense_weight = dense_weight

    def search(
        self,
        query: str,
        top_k: int = 50,
        candidates_per_retriever: int = 100,
        query_tokens: list[str] | None = None,
    ) -> list[RetrievedDoc]:
        """query_tokens — лемматизированный запрос для BM25; query — сырой текст для dense."""
        bm25_hits = self.bm25.search_tokens(
            query_tokens if query_tokens is not None else query.lower().split(),
            top_k=candidates_per_retriever,
        )

        rankings = [[h.doc_id for h in bm25_hits]]
        docs = {h.doc_id: h for h in bm25_hits}

        if self.dense is not None:
            dense_hits = self.dense.search(query, top_k=candidates_per_retriever)
            rankings.append([h.doc_id for h in dense_hits])
            for h in dense_hits:
                docs.setdefault(h.doc_id, h)

        # Взвешенный RRF: вклад dense-ранжирования масштабируется на dense_weight
        scores: dict[str, float] = {}
        weights = [1.0, self.dense_weight]
        for weight, ranking in zip(weights[: len(rankings)], rankings):
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (self.rrf_k + rank)
        fused = sorted(scores.items(), key=lambda x: -x[1])

        results = []
        for doc_id, fused_score in fused[:top_k]:
            base = docs.get(doc_id)
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    text=base.text if base else "",
                    score=fused_score,
                    source="hybrid",
                    metadata=base.metadata if base else None,
                )
            )
        return results