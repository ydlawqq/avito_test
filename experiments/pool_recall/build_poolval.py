"""Новая валидация для pool-экспериментов: чистый срез train.parquet.

build_validation() сортирует все сессии train.parquet по md5-ключу, поэтому
выборка детерминирована: artifacts/validation.parquet (2452 сессии) — это
первые 2452 строк этого порядка. Чтобы новая валидация гарантированно не
пересекалась со старой (и с тем, на чём раньше тюнились бусты), берём
СЛЕДУЮЩИЙ блок: строки [OFFSET : OFFSET+SIZE] md5-порядка.

Выход: artifacts/pool_recall/poolval.parquet — колонки как в validation.parquet
(query_id, search_*, relevant_items).

Запуск:
    python experiments/pool_recall/build_poolval.py                 # 2000 сессий
    python experiments/pool_recall/build_poolval.py --size 500      # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))

import pandas as pd

from src.evaluation.validation import build_validation
from src.pipeline.common import PROJECT_ROOT as ROOT

OFFSET = 2452  # ровно размер старой validation.parquet в md5-порядке


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "pool_recall" / "poolval.parquet"),
    )
    args = parser.parse_args()

    # берём OFFSET+size сессий md5-порядка, отрезаем первые OFFSET
    big = build_validation(
        ROOT / "raw_data" / "train.parquet",
        ROOT / "raw_data" / "benchmark_items.parquet",
        size=OFFSET + args.size,
    )
    val = big.iloc[OFFSET:].reset_index(drop=True)
    assert len(val) == args.size, f"ожидалось {args.size}, получено {len(val)}"

    # страховка от пересечения со старой валидацией
    old = pd.read_parquet(ROOT / "artifacts" / "validation.parquet")
    overlap = set(val["query_id"]) & set(old["query_id"])
    assert not overlap, f"пересечение со старой validation: {len(overlap)} сессий"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    val.to_parquet(out, index=False)

    n_pos = val["relevant_items"].map(len)
    print(f"Сохранён: {out} ({len(val)} сессий)")
    print(f"  релевантных на сессию: mean={n_pos.mean():.2f}, "
          f"median={n_pos.median():.0f}, max={n_pos.max()}")
    print(f"  пересечений со старой validation: 0")


if __name__ == "__main__":
    main()
