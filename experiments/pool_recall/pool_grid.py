"""Грид-эксперименты BM25+бусты: максимизация Recall@N входного пула реранкера.

Замеряется recall сырого BM25-пула (до бустов) и бустнутого пула на валидации
artifacts/pool_recall/poolval.parquet (сессии из train.parquet, вне старой
validation). Релевантность — relevant_items сессии.

Пайплайн продакшна: чистый BM25 по ВСЕМУ корпусу -> сырой топ-RAW_POOL
-> score * (1 + loc_boost * [loc==search_loc]) -> топ-depth (вход XGB).

Грид: title_weight ∈ {1,3,5}; k1 ∈ {0.5,0.9,1.2,1.5,2.0}; b ∈ {0.3,0.5,0.75}
(k1/b мутируются на живом индексе — пересборка корпуса не нужна);
loc_boost ∈ {0.5,1,2,3,4} — считается от тех же скоров, почти бесплатно.

Метрики (macro по сессиям):
    raw@{2000,5000,10000}       — recall сырого пула;
    boost<w>@{1000,2000,5000}   — recall бустнутого пула (w = loc_boost).

Запуск из корня проекта:
    python experiments/pool_recall/pool_grid.py                 # полный грид
    python experiments/pool_recall/pool_grid.py --limit 200     # smoke
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import numpy as np
import pandas as pd

from fast_bm25 import FastBM25, parity_check
from recall_at_k_bm25 import QueryLemmatizer
from src.pipeline.common import PROJECT_ROOT as ROOT, build_bm25, load_config, load_items
from src.retrievers.retrieval import BM25Retriever

RAW_POOL = 10_000
BOOST_DEPTHS = (1000, 2000, 5000)
RAW_DEPTHS = (2000, 5000, 10_000)
K1_GRID = (0.5, 0.9, 1.2, 1.5, 2.0)
B_GRID = (0.3, 0.5, 0.75)
TW_GRID = (1, 3, 5)
LOC_BOOST_GRID = (0.5, 1.0, 2.0, 3.0, 4.0)


def raw_top_indices(scores: np.ndarray, pool: int) -> np.ndarray:
    take = min(pool, len(scores))
    idx = np.argpartition(-scores, take - 1)[:take]
    idx = idx[scores[idx] > 0]
    return idx[np.argsort(-scores[idx], kind="stable")]


def recall_at(pos_idx: set[int], top: np.ndarray, k: int) -> float:
    n_rel = len(pos_idx)
    if n_rel == 0:
        return 0.0
    return len(pos_idx.intersection(top[:k].tolist())) / n_rel


def evaluate_config(fast: FastBM25, item_loc, queries, lemmatizer, pos_lookup, k1, b):
    sums = {f"raw@{k}": 0.0 for k in RAW_DEPTHS}
    for w in LOC_BOOST_GRID:
        for d in BOOST_DEPTHS:
            sums[f"boost{w:g}@{d}"] = 0.0
    n = 0

    for row in queries.itertuples():
        pos = pos_lookup[row.query_id]
        if not pos:
            continue
        tokens = lemmatizer(row.search_query, row.search_infm_params_text)
        scores = fast.scores(tokens, k1, b)
        idx = raw_top_indices(scores, RAW_POOL)

        for k in RAW_DEPTHS:
            sums[f"raw@{k}"] += recall_at(pos, idx, k)

        loc = item_loc[idx]
        s = scores[idx]
        same_loc = loc == row.search_location_id
        for w in LOC_BOOST_GRID:
            boosted = s * (1.0 + w * same_loc)
            order = np.argsort(-boosted, kind="stable")
            bidx = idx[order]
            for d in BOOST_DEPTHS:
                sums[f"boost{w:g}@{d}"] += recall_at(pos, bidx, d)
        n += 1

    return {name: s / max(n, 1) for name, s in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poolval", default=str(ROOT / "artifacts" / "pool_recall" / "poolval.parquet"))
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "pool_recall" / "grid_results.csv"))
    parser.add_argument("--limit", type=int, default=None, help="smoke: первые N сессий")
    args = parser.parse_args()

    t00 = time.time()
    cfg = load_config(ROOT / "configs" / "config.yaml")
    items = load_items(cfg)
    item_loc = items["item_location_id"].to_numpy()

    val = pd.read_parquet(args.poolval)
    if args.limit:
        val = val.head(args.limit)
    print(f"Валидация: {len(val)} сессий", flush=True)

    lemmatizer = QueryLemmatizer()
    item_pos = items["item_id"].reset_index().set_index("item_id")["index"]
    pos_lookup: dict[str, set[int]] = {}
    for row in val.itertuples():
        pos_lookup[row.query_id] = {
            int(item_pos[i]) for i in row.relevant_items if i in item_pos.index
        }

    rows_out = []
    for tw in TW_GRID:
        t0 = time.time()
        cfg_bm25 = dict(cfg["bm25"])
        cfg_bm25["title_weight"] = tw
        retriever = build_bm25(items, cfg_bm25)
        fast = FastBM25(retriever.bm25)
        # паритет с rank_bm25 на первых 20 запросах валидации (k1/b текущего грида)
        check_q = [
            lemmatizer(r.search_query, r.search_infm_params_text)
            for r in val.head(20).itertuples()
        ]
        diffs = [
            parity_check(fast, retriever.bm25, [toks], k1c, bc)
            for k1c, bc in [(K1_GRID[0], B_GRID[0]), (K1_GRID[-1], B_GRID[-1])]
            for toks in check_q
        ]
        print(f"tw={tw}: паритет fast/rank_bm25 max|diff|={max(diffs):.2e}", flush=True)
        print(f"tw={tw}: корпус собран за {time.time() - t0:.0f} c", flush=True)

        for k1, b in itertools.product(K1_GRID, B_GRID):
            t1 = time.time()
            res = evaluate_config(fast, item_loc, val, lemmatizer, pos_lookup, k1, b)
            res.update(title_weight=tw, k1=k1, b=b)
            rows_out.append(res)
            print(f"  tw={tw} k1={k1} b={b}: raw@5000={res['raw@5000']:.4f} "
                  f"raw@10000={res['raw@10000']:.4f} boost2@1000={res['boost2@1000']:.4f} "
                  f"boost2@2000={res['boost2@2000']:.4f} ({time.time() - t1:.0f} c)", flush=True)
            pd.DataFrame(rows_out).to_csv(args.out, index=False)

    df = pd.DataFrame(rows_out)
    df.to_csv(args.out, index=False)
    best = df.sort_values("boost2@2000", ascending=False).head(10)
    print("\n=== ТОП-10 по boost2@2000 ===")
    print(best[["title_weight", "k1", "b", "raw@2000", "raw@5000", "raw@10000",
                "boost0.5@1000", "boost1@1000", "boost2@1000", "boost3@1000",
                "boost4@1000", "boost2@2000", "boost2@5000"]].to_string(index=False))
    print(f"\nВсего {len(df)} конфигураций за {time.time() - t00:.0f} c")


if __name__ == "__main__":
    main()
