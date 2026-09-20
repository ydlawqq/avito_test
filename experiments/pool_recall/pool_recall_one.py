"""Recall пула (входа XGB) для ОДНОГО конфига BM25 на заданной валидации.

Повторяет продакшн-пайплайн: BM25 (tw/k1/b из аргументов) -> сырой топ-pool
-> score * (1 + loc_boost * [loc==search_loc]) -> топ-depth. Печатает recall
сырого и бустнутого пула на глубинах 2000/5000/10000.

Запуск из корня проекта:
    python experiments/pool_recall/pool_recall_one.py \
        --poolval artifacts/validation.parquet --k1 0.9 --b 0.3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/pool_recall"))

import pandas as pd

from fast_bm25 import FastBM25
from pool_grid import evaluate_config
from recall_at_k_bm25 import QueryLemmatizer
from src.pipeline.common import PROJECT_ROOT as ROOT, build_bm25, load_config, load_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poolval", default=str(
        ROOT / "artifacts/pool_recall/poolval.parquet"))
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.3)
    parser.add_argument("--title-weight", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    cfg = load_config(ROOT / "configs" / "config.yaml")
    items = load_items(cfg)
    item_loc = items["item_location_id"].to_numpy()

    val = pd.read_parquet(args.poolval)
    if args.limit:
        val = val.head(args.limit)

    lemmatizer = QueryLemmatizer()
    item_pos = items["item_id"].reset_index().set_index("item_id")["index"]
    pos_lookup = {
        r.query_id: {int(item_pos[i]) for i in r.relevant_items if i in item_pos.index}
        for r in val.itertuples()
    }

    cfg_bm25 = dict(cfg["bm25"])
    cfg_bm25["title_weight"] = args.title_weight
    retriever = build_bm25(items, cfg_bm25)
    fast = FastBM25(retriever.bm25)
    print(f"корпус собран за {time.time() - t0:.0f} c", flush=True)

    res = evaluate_config(
        fast, item_loc, val, lemmatizer, pos_lookup, args.k1, args.b
    )
    print(f"\n== {Path(args.poolval).name}: {len(val)} сессий, "
          f"tw={args.title_weight} k1={args.k1} b={args.b} ==")
    for name in sorted(res):
        print(f"  {name:>16s}: {res[name]:.4f}")
    print(f"({time.time() - t0:.0f} c)")


if __name__ == "__main__":
    main()
