"""Оценка Recall@50 на локальной валидации.

Запуск:
    python scripts/evaluate.py --method bm25   [--limit 500]   # CPU, быстро
    python scripts/evaluate.py --method dense  [--limit 500]   # нужен artifacts/dense/e5.index
    python scripts/evaluate.py --method hybrid [--limit 500]
    python scripts/evaluate.py --method xgb    [--limit 500]   # XGBoost-реранкер
    python scripts/evaluate.py --method xgb_ce [--limit 500]   # XGB + кросс-энкодер

bm25: перебирает варианты конфигурации (описание on/off, вес заголовка,
бусты категории/локации) и печатает Recall@50 каждого — выбор лучшего
варианта основан только на train-данных.

Валидация построена из train.parquet: только запросы, чьи релевантные
объявления есть в корпусе benchmark_items.parquet (см. src/evaluation/validation.py).

xgb: оценка XGBoost-реранкера (BM25 с бустами -> топ-1000 -> XGBRanker ->
топ-50) ровно тем же кодом, что scripts/make_answer.py --method xgb (см.
src/rerank/inference.py), поэтому метрика соответствует формируемому ответу.
По умолчанию считается на validation.parquet: если модель обучалась на датасете,
куда эти сессии вошли в train-часть (artifacts/xgb_rerank/dataset*.parquet),
цифра оптимистична (утечка). Честный режим:
    --dataset artifacts/xgb_rerank/dataset_resampled.parquet --split holdout

xgb_ce: оценка второго слоя реранка кросс-энкодером (BM25 с бустами -> топ-1000
-> XGBRanker -> топ-`--xgb-topk` 150 -> кросс-энкодер -> топ-50). Печатает
таблицу сразу для трёх слоёв на одних и тех же запросах: `xgb+ce`, `xgb`
и `bm25+loc` — видно вклад каждого слоя. Кросс-энкодер считается на GPU
(если доступна) и по умолчанию берёт BAAI/bge-reranker-v2-m3 (cfg.cross_encoder).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))

import pandas as pd
from tqdm import tqdm

from src.evaluation.validation import load_validation, validation_relevance
from src.metrics.recall import mean_recall_at_k
from src.pipeline.common import (
    apply_metadata_boosts,
    artifacts_dir,
    build_bm25,
    lemmatize_query,
    load_config,
    load_items,
)

# Сетка BM25-вариантов для экспериментов (лучший выбираем по валидации).
# На этой данных: 83% релевантных items лежат в локации поиска -> буст
# локации сильно поднимает recall; категория почти вся одна (114) -> буст
# категории не даёт ничего (проверено экспериментально).
BM25_VARIANTS = [
    # (name, use_description, title_weight, category_boost, location_boost, pool)
    ("tw=3 loc 2.0, pool 1000", True, 3, 0.0, 2.0, 1000),
    ("tw=3 loc 2.0, pool 2000", True, 3, 0.0, 2.0, 2000),
    ("tw=3 loc 2.0, pool 5000", True, 3, 0.0, 2.0, 5000),
    ("tw=4 loc 2.0, pool 2000", True, 4, 0.0, 2.0, 2000),
    ("tw=2 loc 2.0, pool 5000", True, 2, 0.0, 2.0, 5000),
]


def subsample_corpus(
    items: pd.DataFrame, validation: pd.DataFrame, n: int, seed: int = 42
) -> pd.DataFrame:
    """Подвыборка корпуса для быстрых/экономных прогонов.

    Все релевантные items валидации всегда включаются в подвыборку —
    иначе метрика будет занижена; добираем случайными дистракторами до n.
    """
    import numpy as np

    relevant = set()
    for rel in validation["relevant_items"]:
        relevant.update(rel)
    rel_part = items[items["item_id"].isin(relevant)]
    rest = items[~items["item_id"].isin(relevant)]
    n_distract = max(0, n - len(rel_part))
    if n_distract < len(rest):
        rest = rest.sample(n_distract, random_state=seed)
    out = pd.concat([rel_part, rest]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"Подвыборка корпуса: {len(out)} объявлений "
        f"(из них релевантных: {len(rel_part)})"
    )
    return out


def evaluate_bm25(
    cfg: dict,
    validation: pd.DataFrame,
    limit: int | None,
    sample_corpus: int | None = None,
) -> None:
    """Перебор вариантов BM25.

    sample_corpus: чтобы не поймать OOM на полном корпусе, можно оценить
    варианты на подвыборке корпуса (все релевантные items всегда включаются).
    Финальные цифры снимать на полном корпусе (sample_corpus=None).
    """
    items = load_items(cfg)
    if sample_corpus:
        items = subsample_corpus(items, validation, sample_corpus)

    # ВАЖНО: метрику считаем только по оцениваемым запросам (queries),
    # иначе mean делится на все запросы валидации и занижается при --limit
    queries = validation.head(limit) if limit else validation
    relevance = validation_relevance(queries)
    k = 500

    qid_to_tokens = {
        row.query_id: lemmatize_query(row.search_query, row.search_infm_params_text)
        for row in queries.itertuples()
    }

    print("\n== BM25: Recall@50 по вариантам ==")
    print(f"{'вариант':38s} {'Recall@50':>10s}")
    import gc

    # Индексы строим ПО ОДНОМУ и освобождаем после варианта — BM25Okapi
    # на полном корпусе держит в памяти ~гигабайты, кэш двух индексов = OOM
    built_keys: set[tuple[bool, int]] = set()
    for vi, (name, use_desc, tw, cat_b, loc_b, pool) in enumerate(BM25_VARIANTS):
        key = (use_desc, tw)
        if key not in built_keys:
            retriever = build_bm25(
                items,
                {"use_description": use_desc, "title_weight": tw, "use_params": True},
            )
            built_keys.add(key)
        # NB: индекс одного key переиспользуется подряд идущими вариантами

        predictions: dict[str, list[str]] = {}
        for row in tqdm(queries.itertuples(), total=len(queries), desc=name, leave=False):
            tokens = qid_to_tokens[row.query_id]
            if cat_b > 0 or loc_b > 0:
                predictions[row.query_id] = apply_metadata_boosts(
                    retriever,
                    tokens,
                    items,
                    top_k=k,
                    search_category=row.search_category,
                    search_location_id=row.search_location_id,
                    category_boost=cat_b,
                    location_boost=loc_b,
                    pool=pool,
                )
            else:
                predictions[row.query_id] = [
                    h.doc_id for h in retriever.search_tokens(tokens, top_k=k)
                ]

        score = mean_recall_at_k(predictions, relevance, k=k)
        print(f"{name:38s} {score:>10.4f}", flush=True)

        # Освобождаем индекс, если следующий вариант использует другой корпус
        next_vi = vi + 1
        if next_vi >= len(BM25_VARIANTS) or (
            BM25_VARIANTS[next_vi][1],
            BM25_VARIANTS[next_vi][2],
        ) != key:
            del retriever
            built_keys.clear()
            gc.collect()


def _load_dense(cfg: dict):
    from src.retrievers.retrieval import DenseRetriever

    index_path = artifacts_dir(cfg) / "dense" / "e5.index"
    if not index_path.exists():
        sys.exit(
            f"Индекс не найден: {index_path}\n"
            "Сначала постройте его: python scripts/build_dense_index.py"
        )
    return DenseRetriever(
        index_path=str(index_path),
        model_name=cfg["dense"]["model_name"],
    )


def evaluate_dense(cfg: dict, validation: pd.DataFrame, limit: int | None) -> None:
    dense = _load_dense(cfg)
    queries = validation.head(limit) if limit else validation
    relevance = validation_relevance(queries)
    k = cfg["recall"]["k"]

    predictions = {}
    for row in tqdm(queries.itertuples(), total=len(queries), desc="dense"):
        # Dense-модель работает на сыром тексте: запрос + фильтры
        query = " ".join(
            x for x in [row.search_query, str(row.search_infm_params_text)] if x
        )
        predictions[row.query_id] = [h.doc_id for h in dense.search(query, top_k=k)]

    print(f"\n== Dense: Recall@50 = {mean_recall_at_k(predictions, relevance, k=k):.4f}")


def evaluate_hybrid(cfg: dict, validation: pd.DataFrame, limit: int | None) -> None:
    from src.retrievers.retrieval import HybridRetriever

    items = load_items(cfg)
    dense = _load_dense(cfg)
    bm25 = build_bm25(items, cfg["bm25"])
    hybrid = HybridRetriever(
        bm25=bm25,
        dense=dense,
        rrf_k=cfg["hybrid"]["rrf_k"],
        dense_weight=cfg["hybrid"]["dense_weight"],
    )

    queries = validation.head(limit) if limit else validation
    relevance = validation_relevance(queries)
    k = cfg["recall"]["k"]
    cpr = cfg["hybrid"]["candidates_per_retriever"]

    predictions = {}
    for row in tqdm(queries.itertuples(), total=len(queries), desc="hybrid"):
        query = " ".join(
            x for x in [row.search_query, str(row.search_infm_params_text)] if x
        )
        tokens = lemmatize_query(row.search_query, row.search_infm_params_text)
        predictions[row.query_id] = [
            h.doc_id
            for h in hybrid.search(
                query, top_k=k, candidates_per_retriever=cpr, query_tokens=tokens
            )
        ]

    print(f"\n== Hybrid: Recall@50 = {mean_recall_at_k(predictions, relevance, k=k):.4f}")

    # Дополнительно: BM25-only на тех же запросах — бенчмарк для сравнения вклада dense
    bm25_only = {}
    for row in queries.itertuples():
        tokens = lemmatize_query(row.search_query, row.search_infm_params_text)
        bm25_only[row.query_id] = [
            h.doc_id for h in bm25.search_tokens(tokens, top_k=k)
        ]
    print(f"   (BM25-only на тех же запросах: {mean_recall_at_k(bm25_only, relevance, k=k):.4f})")


def _select_eval_set(
    cfg: dict,
    validation: pd.DataFrame,
    limit: int | None,
    dataset: str | None,
    split: str,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Выбрать (queries, relevance) для оценки.

    * dataset=None  -> validation.parquet (утечка, если модель на нём обучалась);
    * dataset!=None -> честный сплит датасета реранкера, атрибуты запросов
      восстанавливаются из train.parquet.
    """
    if dataset:
        relevance = _dataset_relevance(cfg, dataset, split)
        print(f"Честная оценка: {dataset} split={split} "
              f"({len(relevance)} сессий с позитивами)")
        qids = list(relevance)
        if limit:
            qids = qids[:limit]
        queries = _sessions_for_qids(cfg, qids)
        relevance = {q: relevance[q] for q in queries["query_id"]}
        return queries, relevance
    queries = validation.head(limit) if limit else validation
    return queries, validation_relevance(queries)


def evaluate_xgb(
    cfg: dict,
    validation: pd.DataFrame,
    limit: int | None,
    model_path: str | None = None,
    categories_path: str | None = None,
    depth: int | None = None,
    dataset: str | None = None,
    split: str = "holdout",
    ks: list[int] | None = None,
) -> None:
    """Recall@k XGBoost-реранкера (BM25 с бустами -> топ-`depth` -> XGB -> топ-50).

    Использует ровно тот же код, что scripts/make_answer.py --method xgb
    (src/rerank/inference.py) — метрика соответствует формируемому ответу.

    Режимы оценки:
      * по умолчанию — на validation.parquet. ВНИМАНИЕ: если модель обучалась на
        датасете, куда эти сессии вошли в train-часть (dataset.parquet,
        dataset_resampled.parquet), цифра оптимистична из-за утечки;
      * --dataset <parquet> --split holdout — честный режим: сессии из указанного
        сплита датасета (например holdout) не участвовали ни в обучении, ни в
        early stopping. Атрибуты запросов восстанавливаются из train.parquet.

    ks: список k для метрики (по умолчанию cfg["recall"]["k"] = 50). Для k > 50
    реранкер возвращает больше кандидатов (max_answer = max(ks)), но не больше
    глубины пула `depth`. Печатается таблица Recall@k для реранкера и бейзлайна.
    """
    from src.rerank.inference import rerank_xgb

    xcfg = cfg.get("xgb_rerank", {})
    model_path = model_path or str(
        PROJECT_ROOT / xcfg.get("model", "artifacts/xgb_rerank/model.json"))
    categories_path = categories_path or str(
        PROJECT_ROOT / xcfg.get("categories", "artifacts/xgb_rerank/cat_categories.json"))
    depth = depth or int(xcfg.get("depth", 1000))
    ks = sorted(ks or [int(cfg["recall"]["k"])])
    max_answer = min(ks[-1], depth)
    if ks[-1] > depth:
        print(f"ВНИМАНИЕ: k={ks[-1]} больше глубины пула depth={depth} — "
              f"оцениваем максимум по {depth} (пул не содержит большего топа)")
        ks = [k for k in ks if k <= depth] or [depth]

    queries, relevance = _select_eval_set(cfg, validation, limit, dataset, split)

    print(f"Запросов для оценки: {len(queries)}")
    predictions, baseline = rerank_xgb(
        cfg, queries, model_path, categories_path, depth,
        collect_baseline=True, max_answer=max_answer,
    )

    print(f"\n== XGB-реранкер (depth={depth}), кандидатов отдаёт: {max_answer} ==")
    print(f"{'k':>5} | {'xgb':>7} | {'bm25+loc':>8} | {'прирост':>8}")
    for k in ks:
        r_xgb = mean_recall_at_k(predictions, relevance, k=k)
        r_base = mean_recall_at_k(baseline, relevance, k=k)
        print(f"{k:>5} | {r_xgb:>7.4f} | {r_base:>8.4f} | {r_xgb - r_base:>+8.4f}")

def evaluate_xgb_ce(
    cfg: dict,
    validation: pd.DataFrame,
    limit: int | None,
    model_path: str | None = None,
    categories_path: str | None = None,
    depth: int | None = None,
    dataset: str | None = None,
    split: str = "holdout",
    ks: list[int] | None = None,
    xgb_topk: int | None = None,
    ce_model: str | None = None,
    ce_device: str | None = None,
    ce_max_length: int | None = None,
    ce_batch_size: int | None = None,
) -> None:
    """Recall@k связки «XGB -> топ-`xgb_topk` -> кросс-энкодер -> топ-max(k)».

    Новый второй слой реранка: после XGBoost-ранжировки топ-1000 объявлений
    берётся только топ-`xgb_topk` (по умолчанию 150, cfg.cross_encoder.xgb_topk),
    они переранжируются кросс-энкодером, из его порядка берётся финальный топ-50.

    Печатается таблица для трёх слоёв на ОДНИХ И ТЕХ ЖЕ запросах:
        bm25+loc — бустнутый BM25 (текущий бейзлайн, голова того же пула);
        xgb      — одиночный XGBRanker (топ-1000 -> модель);
        xgb+ce   — новый слой с кросс-энкодером.
    Колонки `ce-xgb` / `ce-bm25` показывают прирост нового слоя.

    Режимы оценки (как в evaluate_xgb):
      * по умолчанию — validation.parquet (с утечкой, если XGB/CE обучались на
        этих сессиях);
      * --dataset <parquet> --split holdout — честный режим без утечки.
    """
    from src.rerank.inference import rerank_xgb_ce

    xcfg = cfg.get("xgb_rerank", {})
    ccfg = cfg.get("cross_encoder", {}) or {}
    model_path = model_path or str(
        PROJECT_ROOT / xcfg.get("model", "artifacts/xgb_rerank/model.json"))
    categories_path = categories_path or str(
        PROJECT_ROOT / xcfg.get("categories", "artifacts/xgb_rerank/cat_categories.json"))
    depth = depth or int(xcfg.get("depth", 1000))
    xgb_topk = int(xgb_topk if xgb_topk is not None
                   else ccfg.get("xgb_topk", 150))
    # кросс-энкодер видит не больше `xgb_topk` кандидатов, а те — не больше depth
    ce_pool = min(xgb_topk, depth)

    ks = sorted(ks or [int(cfg["recall"]["k"])])
    if ks[-1] > ce_pool:
        print(f"ВНИМАНИЕ: k={ks[-1]} больше входа кросс-энкодера xgb_topk={ce_pool} "
              f"— оцениваем максимум по {ce_pool}")
        ks = [k for k in ks if k <= ce_pool] or [ce_pool]
    max_answer = ks[-1]

    queries, relevance = _select_eval_set(cfg, validation, limit, dataset, split)
    print(f"Запросов для оценки: {len(queries)} | xgb_topk={xgb_topk} | "
          f"кросс-энкодер: {ce_model or ccfg.get('model', 'BAAI/bge-reranker-v2-m3')}")

    predictions, baseline, xgb_only = rerank_xgb_ce(
        cfg, queries, model_path, categories_path, depth,
        xgb_topk=xgb_topk, ce_model=ce_model, ce_device=ce_device,
        ce_max_length=ce_max_length, ce_batch_size=ce_batch_size,
        collect_baseline=True, max_answer=max_answer,
    )

    print(f"\n== XGB -> топ-{xgb_topk} -> кросс-энкодер (depth={depth}), "
          f"отдаёт топ-{max_answer} ==")
    print(f"{'k':>5} | {'xgb+ce':>7} | {'xgb':>7} | {'bm25+loc':>8} | "
          f"{'ce-xgb':>7} | {'ce-bm25':>8}")
    for k in ks:
        r_ce = mean_recall_at_k(predictions, relevance, k=k)
        r_xgb = mean_recall_at_k(xgb_only, relevance, k=k)
        r_base = mean_recall_at_k(baseline, relevance, k=k)
        print(f"{k:>5} | {r_ce:>7.4f} | {r_xgb:>7.4f} | {r_base:>8.4f} | "
              f"{r_ce - r_xgb:>+7.4f} | {r_ce - r_base:>+8.4f}")


def _parse_ks(text: str | None, cfg: dict) -> list[int]:
    """Список k для Recall@k: из --k ("50,100,150") или cfg.recall.k."""
    if not text:
        return [int(cfg["recall"]["k"])]
    ks = sorted({int(x) for x in text.replace(",", " ").split()})
    if not ks:
        sys.exit(f"Не удалось разобрать --k: {text!r}")
    return ks


def _dataset_relevance(cfg: dict, dataset: str, split: str) -> dict[str, set[str]]:
    """{query_id: {релевантные item_id}} по сплиту датасета реранкера."""
    ds = pd.read_parquet(dataset, columns=["query_id", "item_id", "label", "split"])
    ds = ds[ds["split"] == split]
    if ds.empty:
        sys.exit(f"В {dataset} нет строк со split={split!r}")
    rel = {
        q: set(g.loc[g["label"] > 0, "item_id"])
        for q, g in ds.groupby("query_id")
    }
    return {q: items for q, items in rel.items() if items}


def _sessions_for_qids(cfg: dict, qids: list[str]) -> pd.DataFrame:
    """Атрибуты запросов (search_*) для нужных query_id из train.parquet.

    Датасет реранкера хранит только query_id (без текста запроса), поэтому
    признаки запроса восстанавливаем из train.parquet тем же способом, что и
    src/evaluation/validation.py (тот же фильтр по корпусу и та же формула
    query_id = md5(ключ запроса)[:16]), но БЕЗ построения всех 26.5k сессий:
    сначала считаем ключи уникальных запросов, затем оставляем нужные —
    так пик памяти кратно ниже (в датасете есть списки релевантных item_id).
    """
    from hashlib import md5

    from src.evaluation.validation import QUERY_KEY_COLUMNS

    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_data_dir"]
    corpus_ids = set(
        pd.read_parquet(artifacts_dir(cfg) / "items_processed.parquet",
                        columns=["item_id"])["item_id"]
    )
    train = pd.read_parquet(raw_dir / "train.parquet",
                            columns=QUERY_KEY_COLUMNS + ["item_id"])
    train = train[train["item_id"].isin(corpus_ids)]
    # уникальных ключей запросов ~26.5k против 497k строк train.parquet
    keys = train[QUERY_KEY_COLUMNS].dropna().drop_duplicates(
        subset=QUERY_KEY_COLUMNS, ignore_index=True)
    keys["query_id"] = [
        md5("\x1f".join(str(v) for v in vals).encode("utf-8")).hexdigest()[:16]
        for vals in keys[QUERY_KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    sess = keys[keys["query_id"].isin(set(qids))].reset_index(drop=True)
    missing = set(qids) - set(sess["query_id"])
    if missing:
        print(f"ВНИМАНИЕ: атрибуты не найдены для {len(missing)} query_id")
    return sess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid", "xgb", "xgb_ce"],
                        required=True)
    parser.add_argument("--limit", type=int, default=None, help="число запросов валидации")
    parser.add_argument(
        "--sample-corpus",
        type=int,
        default=None,
        help="оценить на подвыборке корпуса из N объявлений (анти-OOM режим; "
        "релевантные items всегда включаются). Финальные цифры — без флага.",
    )
    parser.add_argument("--model", default=None,
                        help="model.json XGBRanker (методы xgb / xgb_ce)")
    parser.add_argument("--categories", default=None,
                        help="cat_categories.json (методы xgb / xgb_ce)")
    parser.add_argument("--depth", type=int, default=None,
                        help="глубина переранжированного пула (методы xgb / xgb_ce)")
    parser.add_argument("--dataset", default=None,
                        help="датасет реранкера для честной оценки без утечки "
                        "(методы xgb / xgb_ce; напр. artifacts/xgb_rerank/dataset_resampled.parquet)")
    parser.add_argument("--k", default=None,
                        help="k для Recall@k через запятую/пробел (методы xgb / xgb_ce; "
                        "по умолчанию cfg.recall.k=50). Пример: --k 50,100,150")
    parser.add_argument("--split", default="holdout",
                        help="сплит датасета для --dataset: holdout/eval/train "
                        "(методы xgb / xgb_ce; holdout — сессии, не участвовавшие в обучении)")
    parser.add_argument("--xgb-topk", type=int, default=None,
                        help="сколько кандидатов XGB отдаёт кросс-энкодеру "
                        "(метод xgb_ce; по умолчанию cfg.cross_encoder.xgb_topk=150)")
    parser.add_argument("--ce-model", default=None,
                        help="имя/путь кросс-энкодера (метод xgb_ce; "
                        "по умолчанию cfg.cross_encoder.model=BAAI/bge-reranker-v2-m3)")
    parser.add_argument("--ce-device", default=None,
                        help="device кросс-энкодера: cuda / cpu (метод xgb_ce; "
                        "по умолчанию автоопределение)")
    parser.add_argument("--ce-max-length", type=int, default=None,
                        help="макс. токенов пары для кросс-энкодера (метод xgb_ce)")
    parser.add_argument("--ce-batch-size", type=int, default=None,
                        help="batch size кросс-энкодера (метод xgb_ce)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    validation = load_validation(artifacts_dir(cfg) / "validation.parquet")
    print(f"Валидация: {len(validation)} запросов (limit={args.limit or 'all'})")

    if args.method == "bm25":
        evaluate_bm25(cfg, validation, args.limit, sample_corpus=args.sample_corpus)
    elif args.method == "dense":
        evaluate_dense(cfg, validation, args.limit)
    elif args.method == "xgb":
        evaluate_xgb(cfg, validation, args.limit, model_path=args.model,
                     categories_path=args.categories, depth=args.depth,
                     dataset=args.dataset, split=args.split,
                     ks=_parse_ks(args.k, cfg))
    elif args.method == "xgb_ce":
        evaluate_xgb_ce(cfg, validation, args.limit, model_path=args.model,
                        categories_path=args.categories, depth=args.depth,
                        dataset=args.dataset, split=args.split,
                        ks=_parse_ks(args.k, cfg), xgb_topk=args.xgb_topk,
                        ce_model=args.ce_model, ce_device=args.ce_device,
                        ce_max_length=args.ce_max_length,
                        ce_batch_size=args.ce_batch_size)
    else:
        evaluate_hybrid(cfg, validation, args.limit)


if __name__ == "__main__":
    main()
