"""Векторная добивка готового dataset.parquet новыми v2-признаками.

Не пересобирает BM25-пулы: старые 30 фич остаются как были, добавляются
12 фич из V2_COLUMNS (см. solution/src/rerank/xgb_features.py):
гео-расстояние, парная статистика локаций, позитивные статистики
train-split, jaccard-пересечения. Формулы ОДНИ И ТЕ ЖЕ, что в
build_features() — используется PairStats + haversine_km из модуля.

Паритет проверяется режимом --check N: для N сессий пересчитываются пулы
через BM25 и build_features, значения сравниваются с векторным расчётом
(compute_v2) по каждой фиче (max abs diff).

Запуск из корня проекта:
    python solution/xgb_rerank/augment_dataset.py             # полный прогон
    python solution/xgb_rerank/augment_dataset.py --check 12  # паритет-чек
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "solution"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.preprocessing import get_stopwords
from src.pipeline.common import load_config, lemmatize_query
from src.evaluation.validation import build_validation
from src.rerank.xgb_features import (
    NUMERIC_FEATURES,
    V2_COLUMNS,
    PairStats,
    haversine_km,
    set_pair_stats,
)


def build_session_meta() -> pd.DataFrame:
    """query_id -> qtext, qset_len, params_nonempty (по ВСЕМ сессиям)."""
    sessions = build_validation(
        PROJECT_ROOT / "raw_data/train.parquet",
        PROJECT_ROOT / "artifacts/items_processed.parquet",
        size=10**9,
    )
    sw = get_stopwords()
    meta = pd.DataFrame({
        "query_id": sessions["query_id"].to_numpy(),
        "qtext": sessions["search_query"].astype(str).to_numpy(),
        "search_location_id": sessions["search_location_id"].to_numpy(),
    })
    # qset_len: len(frozenset(леммы запроса+фильтров)) — как в build_features
    qset_len = np.empty(len(meta), dtype=np.int64)
    for i, (q, p) in enumerate(zip(
        sessions["search_query"].astype(str),
        sessions["search_infm_params_text"].astype(str),
    )):
        qset_len[i] = len(frozenset(lemmatize_query(q, p, sw)))
    meta["qset_len"] = qset_len
    meta["params_nonempty"] = [
        1.0 if str(p).strip() else 0.0
        for p in sessions["search_infm_params_text"].astype(str)
    ]
    return meta


def make_items_side(cfg: dict) -> pd.DataFrame:
    """item_id -> локация, microcat, координаты (для векторной добивки)."""
    it = pd.read_parquet(
        PROJECT_ROOT / cfg["paths"]["raw_data_dir"] / "benchmark_items.parquet",
        columns=[
            "item_id", "item_location_id", "item_microcat_id",
            "item_latitude", "item_longitude",
        ],
    )
    it = it.rename(columns={"item_latitude": "lat", "item_longitude": "lon"})
    it["lat"] = pd.to_numeric(it["lat"], errors="coerce")
    it["lon"] = pd.to_numeric(it["lon"], errors="coerce")
    return it


def compute_v2(chunk: pd.DataFrame, ps: PairStats, smeta: pd.DataFrame,
               items_side: pd.DataFrame) -> pd.DataFrame:
    """Векторный расчёт V2_COLUMNS для чанка строк датасета.

    chunk: колонки query_id, item_id, search_loc, title_overlap,
           params_overlap, title_len, params_len.
    items_side: item_id -> item_location_id, item_microcat_id, lat, lon.
    smeta: query_id -> qtext, qset_len, params_nonempty.
    """
    n = len(chunk)
    df = chunk.merge(items_side, on="item_id", how="left")
    df = df.merge(smeta, on="query_id", how="left")

    out = pd.DataFrame(index=chunk.index)

    # --- гео: расстояние до центроида search_loc ---
    clat = df["search_location_id"].map(ps.cent_lat).to_numpy(dtype=np.float64)
    clon = df["search_location_id"].map(ps.cent_lon).to_numpy(dtype=np.float64)
    ilat = df["lat"].to_numpy(dtype=np.float64)
    ilon = df["lon"].to_numpy(dtype=np.float64)
    ok = ~(np.isnan(clat) | np.isnan(clon) | np.isnan(ilat) | np.isnan(ilon))
    geo = np.full(n, np.nan)
    geo[ok] = haversine_km(ilat[ok], ilon[ok], clat[ok], clon[ok])
    out["geo_dist_km"] = geo.astype(np.float32)
    glog = np.full(n, np.nan)
    glog[ok] = np.log1p(geo[ok])
    out["geo_dist_log"] = glog.astype(np.float32)

    # --- парная статистика локаций P(item_loc | search_loc) ---
    mi = pd.MultiIndex.from_arrays(
        [df["search_location_id"].to_numpy(), df["item_location_id"].to_numpy()]
    )
    known_s = df["search_location_id"].isin(ps.cent_lat.index).to_numpy()
    cnt = ps.loc_pairs_cnt.reindex(mi).to_numpy(dtype=np.float64)
    prob = ps.loc_pairs_prob.reindex(mi).to_numpy(dtype=np.float64)
    prob = np.where(known_s & np.isnan(prob), 0.0, prob)
    out["loc_pair_prob"] = prob.astype(np.float32)
    out["loc_pair_cnt_log"] = np.log1p(np.clip(np.nan_to_num(cnt), 0, None)).astype(np.float32)

    # --- позитивные статистики (все строки train, кроме eval/holdout-позитивов) ---
    ipc = df["item_id"].map(ps.item_pos).to_numpy(dtype=np.float64)
    out["item_pos_cnt_log"] = np.log1p(np.nan_to_num(ipc, nan=0.0)).astype(np.float32)
    out["microcat_pos_freq"] = (
        df["item_microcat_id"].map(ps.microcat_pos).to_numpy(dtype=np.float32)
    )
    lmf = ps.loc_microcat_freq.reindex(pd.MultiIndex.from_arrays([
        df["search_location_id"].to_numpy(), df["item_microcat_id"].to_numpy(),
    ])).to_numpy(dtype=np.float64)
    out["loc_microcat_freq"] = lmf.astype(np.float32)

    # --- статистики по тексту запроса ---
    qseen = df["qtext"].isin(ps.qtext_cnt.index).to_numpy()
    qmf = ps.qtext_microcat_freq.reindex(pd.MultiIndex.from_arrays([
        df["qtext"].to_numpy(), df["item_microcat_id"].to_numpy(),
    ])).to_numpy(dtype=np.float64)
    qmf = np.where(qseen, np.nan_to_num(qmf, nan=0.0), np.nan)
    out["qtext_microcat_freq"] = qmf.astype(np.float32)
    # qtext_item_cnt_log — признак model_v2 (log1p(count(text, item)) из
    # train-позитивов; текст запроса не встречался в train -> 0)
    qic = ps.qtext_item_cnt.reindex(pd.MultiIndex.from_arrays([
        df["qtext"].to_numpy(), df["item_id"].to_numpy(),
    ])).to_numpy(dtype=np.float64)
    qic = np.where(qseen, np.nan_to_num(qic, nan=0.0), 0.0)
    out["qtext_item_cnt_log"] = np.log1p(np.clip(qic, 0, None)).astype(np.float32)

    # --- jaccard по множествам лемм ---
    qsl = df["qset_len"].to_numpy(dtype=np.int64)
    t_ov = df["title_overlap"].to_numpy(dtype=np.float64)
    p_ov = df["params_overlap"].to_numpy(dtype=np.float64)
    u_t = qsl + df["title_len"].to_numpy(dtype=np.int64) - t_ov
    u_p = qsl + df["params_len"].to_numpy(dtype=np.int64) - p_ov
    out["title_jaccard"] = np.divide(t_ov, u_t, out=np.zeros(n), where=u_t > 0).astype(np.float32)
    out["params_jaccard"] = np.divide(p_ov, u_p, out=np.zeros(n), where=u_p > 0).astype(np.float32)
    out["search_params_nonempty"] = df["params_nonempty"].to_numpy(dtype=np.float32)

    return out[V2_COLUMNS]


def run_augment(cfg: dict, src: Path, dst: Path, chunk_rows: int,

                stats_dir: Path) -> None:
    t0 = time.time()
    ps = PairStats(stats_dir)
    set_pair_stats(ps)
    smeta = build_session_meta()
    items_side = make_items_side(cfg)
    print(f"session meta: {len(smeta)}, items_side: {len(items_side)} "
          f"({time.time()-t0:.0f} c)", flush=True)

    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    base_cols = list(schema.names)
    fields_by_name = {f.name: f for f in schema}
    # v2-колонки могут УЖЕ быть в источнике (build_dataset с NUMERIC_FEATURES,
    # включающим v2, но без загруженных pair stats -> там NaN). Тогда не
    # вставляем дубликаты, а перезаписываем значения compute_v2 на месте.
    overwrite_v2 = all(c in base_cols for c in V2_COLUMNS)
    if overwrite_v2:
        print("V2-колонки уже есть в источнике — перезапись значений compute_v2")
        target_schema = schema
    else:
        insert_at = base_cols.index("microcat_id")  # v2-фичи перед категориальными
        target_schema = pa.schema(
            [fields_by_name[c] for c in base_cols[:insert_at]]
            + [pa.field(c, pa.float32()) for c in V2_COLUMNS]
            + [fields_by_name[c] for c in base_cols[insert_at:]]
        )
    writer = pq.ParquetWriter(dst, target_schema, compression="zstd")

    rows_done = 0
    for batch in pf.iter_batches(batch_size=chunk_rows):
        df = batch.to_pandas()
        v2 = compute_v2(df, ps, smeta, items_side)
        if overwrite_v2:
            out = df.copy()
            for c in V2_COLUMNS:
                out[c] = v2[c].to_numpy()
        else:
            out = pd.concat(
                [df[base_cols[:insert_at]], v2, df[base_cols[insert_at:]]], axis=1
            )
        table = pa.Table.from_pandas(out, preserve_index=False)
        if table.schema != target_schema:
            table = table.cast(target_schema)
        writer.write_table(table)
        rows_done += len(df)
        print(f"  {rows_done:,} строк ({time.time()-t0:.0f} c)", flush=True)
    writer.close()
    print(f"Сохранён: {dst} ({rows_done:,} строк, {time.time()-t0:.0f} c)")


def run_check(cfg: dict, src: Path, stats_dir: Path, n_sessions: int) -> None:

    """Паритет: build_features (через BM25-пул) vs векторный compute_v2."""
    from src.pipeline.common import build_bm25
    from src.rerank.xgb_features import (
        ItemSideIndex, build_features, boosted_pool, prepare_item_side,
    )

    ps = PairStats(stats_dir)
    set_pair_stats(ps)
    smeta = build_session_meta()
    items_side = make_items_side(cfg)

    print("Строю BM25-корпус для паритет-чека ...", flush=True)
    items = prepare_item_side(cfg)
    retriever = build_bm25(items, cfg["bm25"])
    item_index = ItemSideIndex(items)
    b = cfg["bm25"]

    ds = pd.read_parquet(src, columns=[
        "query_id", "item_id", "label", "search_loc",
        "title_overlap", "params_overlap", "title_len", "params_len",
    ])
    sessions = build_validation(
        PROJECT_ROOT / "raw_data/train.parquet",
        PROJECT_ROOT / "artifacts/items_processed.parquet",
        size=10**9,
    )
    sess_by_qid = sessions.set_index("query_id")
    qids_all = ds["query_id"].drop_duplicates().tolist()
    # приоритет eval-сессиям (полные пулы, строки глубже 50)
    n_train = int(len(sessions) * 0.7)
    pos_of = {q: i for i, q in enumerate(sessions["query_id"])}
    qids_all = [q for q in qids_all if pos_of.get(q, -1) >= n_train] + [
        q for q in qids_all if pos_of.get(q, -1) < n_train
    ]

    n = 0
    diffs: dict[str, list[float]] = {c: [] for c in V2_COLUMNS}
    for qid in qids_all:
        qrow = sess_by_qid.loc[qid]
        tokens = lemmatize_query(qrow.search_query, qrow.search_infm_params_text)
        corpus_idx, raw_s, boost_s, raw_r = boosted_pool(
            retriever, tokens, item_index,
            qrow.search_location_id, qrow.search_category,
            depth=int(cfg.get("xgb_rerank", {}).get("depth", 1000)),
            pool=int(b.get("boost_pool", 5000)),
            loc_boost=float(b.get("location_boost", 2.0)),
            cat_boost=float(b.get("category_boost", 0.0)),
        )
        if len(corpus_idx) == 0:
            continue
        X, _, _ = build_features(
            qrow, corpus_idx, raw_s, boost_s, raw_r, tokens, item_index
        )
        feat = pd.DataFrame(X, columns=NUMERIC_FEATURES)
        feat["item_id"] = item_index.doc_ids[corpus_idx]
        ds_sess = ds[ds.query_id == qid].drop(columns=["query_id"]).set_index("item_id")
        feat = feat.set_index("item_id").join(ds_sess, how="inner", rsuffix="_ds")
        if feat.empty:
            continue
        # векторный пересчёт v2 ровно как в run_augment
        chunk = feat.reset_index()[[
            "item_id", "search_loc", "title_overlap", "params_overlap",
            "title_len", "params_len",
        ]]
        chunk["query_id"] = qid
        v2 = compute_v2(chunk, ps, smeta, items_side)
        for c in V2_COLUMNS:
            left = feat[c].to_numpy(dtype=np.float64)    # build_features
            right = v2[c].to_numpy(dtype=np.float64)     # compute_v2
            both_nan = np.isnan(left) & np.isnan(right)
            d = np.abs(left - right)
            d = d[~np.isnan(d) & ~both_nan]
            if len(d):
                diffs[c].append(float(d.max()))
            if int((np.isnan(left) != np.isnan(right)).sum()):
                diffs[c].append(float("inf"))
        n += 1
        if n % 2 == 0:
            print(f"  сессий проверено: {n}", flush=True)
        if n >= n_sessions:
            break

    print("\n== Паритет build_features vs compute_v2 (max abs diff) ==")
    bad = 0
    for c in V2_COLUMNS:
        mx = max(diffs[c]) if diffs[c] else float("nan")
        flag = "OK" if (np.isnan(mx) or mx < 1e-3) else "MISMATCH"
        bad += flag == "MISMATCH"
        print(f"  {c:>24s}: {mx:.6f} {flag}")
    print(f"\nсессий: {n}, колонок с расхождением: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--src", default=str(
        PROJECT_ROOT / "artifacts" / "xgb_rerank" / "dataset.parquet"))
    parser.add_argument("--out", default=str(
        PROJECT_ROOT / "artifacts" / "xgb_rerank" / "dataset_v2.parquet"))
    parser.add_argument("--stats-dir", default=str(
        PROJECT_ROOT / "artifacts" / "xgb_rerank" / "pair_stats"))
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--check", type=int, default=0,
                        help="режим паритета: N сессий")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.check:
        run_check(cfg, Path(args.src), Path(args.stats_dir), args.check)
    else:
        run_augment(cfg, Path(args.src), Path(args.out), args.chunk_rows,
                    Path(args.stats_dir))


if __name__ == "__main__":
    main()
