"""Кросс-энкодер (реранкер второго уровня) поверх топ-k XGBoost-ранжировки.

Полный пайплайн нового слоя:
    BM25 + бусты локации/категории
        -> сырой пул top-`boost_pool` (5000)
        -> переранжированный топ-`depth` (1000)
        -> XGBRanker
        -> топ-`xgb_topk` (150) кандидатов
        -> кросс-энкодер (по умолчанию BAAI/bge-reranker-v2-m3)
        -> финальный топ-50.

Зачем второй реранкер:
    * градиентный бустинг работает на ~30 табличных признаках пары
      (пересечения лемм, BM25-скоры, метаданные) и «не читает» тексты целиком;
    * кросс-энкодер оценивает пару (запрос, объявление) СОВМЕСТНО механизмом
      внимания — это точнее би-энкодера и табличных фич, но дороже: одну пару
      нужно прогнать через трансформер;
    * поэтому кросс-энкодер применяется только к короткому списку (150 пар на
      запрос вместо ~1000-5000), что делает слой вычислительно приемлемым.

Используется open-source модель (кэшируется HuggingFace):
    BAAI/bge-reranker-v2-m3 — multilingual cross-encoder (ru входит в обучение),
    согласована с dense-моделью BAAI/bge-m3 из cfg["dense"].

Модуль не зависит от xgboost/BM25 — принимает готовые тексты пар, поэтому его
можно использовать и в других слоях (например, переранжировать dense-пул).
"""

from __future__ import annotations

import numpy as np

# Модель по умолчанию (мультиязычный кросс-энкодер, поддерживает русский).
DEFAULT_CE_MODEL = "BAAI/bge-reranker-v2-m3"


def build_query_text(search_query, search_infm_params_text=None) -> str:
    """Текст запроса для кросс-энкодера: search_query + фильтры.

    Ровно те признаки запроса, что доступны в benchmark_queries.parquet.
    Пустые значения и строковые 'nan' (артефакт pandas) отбрасываются.
    """
    query = "" if search_query is None else str(search_query).strip()
    params = "" if search_infm_params_text is None else str(search_infm_params_text).strip()
    if params.lower() == "nan":
        params = ""
    return " ".join(x for x in (query, params) if x)


def build_item_text(
    title,
    params,
    description,
    desc_max_chars: int = 600,
) -> str:
    """Текст объявления для кросс-энкодера: заголовок + параметры + описание.

    Заголовок — главный сигнал соответствия короткому запросу, поэтому стоит
    первым. Описание усекается (`desc_max_chars`) — на 150 кандидатов на запрос
    хвост описания почти не влияет, а время инференса растёт линейно по длине.
    """
    parts = [
        str(title).strip() if title is not None else "",
        str(params).strip() if params is not None else "",
    ]
    if description is not None:
        parts.append(str(description).strip()[:desc_max_chars])
    return ". ".join(p for p in parts if p)


def build_item_texts(items, desc_max_chars: int = 600) -> list[str]:
    """Список текстов объявлений В ПОРЯДКЕ корпуса (индексы = индексы корпуса).

    items — DataFrame с колонками title_raw/params_raw/desc_raw
    (artifacts/items_processed.parquet, см. scripts/prepare_data.py).
    Порядок критичен: corpus_idx из boosted_pool индексирует именно этот список.
    """
    return [
        build_item_text(t, p, d, desc_max_chars)
        for t, p, d in zip(items["title_raw"], items["params_raw"], items["desc_raw"])
    ]


class CrossEncoderReranker:
    """Тонкая обёртка sentence_transformers.CrossEncoder для реранка кандидатов.

    Применение внутри запроса:
        order, scores = reranker.order(query_text, [doc_text, ...])
        top50 = [docs[i] for i in order[:50]]

    Args:
        model_name: имя/путь модели (по умолчанию DEFAULT_CE_MODEL).
        device: 'cuda' / 'cpu'; None -> автоопределение (cuda, если доступна).
        max_length: максимум токенов пары (обрезка длинных описаний).
        batch_size: размер батча при скоринге пар (память/скорость).
        local_files_only: не ходить в сеть (использовать локальный кэш HF).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CE_MODEL,
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        local_files_only: bool = False,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=self.max_length,
            local_files_only=local_files_only,
        )
        print(
            f"Кросс-энкодер загружен: {model_name} | device={self.model.device} | "
            f"max_length={self.max_length} | batch_size={self.batch_size}"
        )

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Скоры готовых пар (query, doc) — форма (len(pairs),), float32.

        Для bge-reranker-v2-m3 это один логит релевантности; для ранжирования
        важен только порядок (монотонные преобразования вроде sigmoid его не
        меняют), поэтому активацию не применяем.
        """
        if not pairs:
            return np.empty(0, dtype=np.float32)
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(pairs):
            raise RuntimeError(
                f"Кросс-энкодер вернул {scores.shape[0]} скоров на {len(pairs)} пар"
            )
        return scores

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Скоры пар (query, doc) в порядке docs."""
        return self.score_pairs([(query, d) for d in docs])

    def order(
        self,
        query: str,
        docs: list[str],
        top_k: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Порядок документов best-first (индексы в docs) + их скоры.

        Сортировка stable: документы с равным скором сохраняют входной порядок
        (т.е. порядок, в котором их отдал XGBoost-реранкер) — детерминизм и
        преемственность ранжирования.
        """
        scores = self.score(query, docs)
        order = np.argsort(-scores, kind="stable")
        if top_k is not None:
            order = order[:top_k]
        return order, scores
