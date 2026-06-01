"""Synthetic performance check for the snapshot-delta update model.

The benchmark is intentionally small and local.  It is meant to catch obvious
scaling regressions in the reference update layer, not to report paper numbers.
Use the production CUDA runners for final 1B experiments.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from dynamic_updates import ResidualPQEncoder, SnapshotDeltaIndex, build_snapshot


def elapsed_ms(fn):
    t0 = time.perf_counter()
    value = fn()
    t1 = time.perf_counter()
    return value, (t1 - t0) * 1000.0


def make_clustered_data(n: int, dim: int, nlist: int, seed: int):
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(nlist, dim)).astype(np.float32) * 4.0
    assign = rng.integers(0, nlist, size=n)
    vectors = centroids[assign] + rng.normal(size=(n, dim)).astype(np.float32) * 0.6
    ids = np.arange(n, dtype=np.int64)
    return ids, vectors, centroids


def run(args: argparse.Namespace) -> List[Dict[str, float]]:
    ids, vectors, centroids = make_clustered_data(args.base_n, args.dim, args.nlist, args.seed)
    base = build_snapshot(ids, vectors, centroids)
    pq = ResidualPQEncoder.train(base.vectors, base.centroids, base.assignments, m=args.pq_m, ksub=16, seed=5)
    base = build_snapshot(ids, vectors, centroids, pq)
    service = SnapshotDeltaIndex(base, pq_encoder=pq, delta_ratio_threshold=0.05, delete_ratio_threshold=0.10)

    rng = np.random.default_rng(args.seed + 1)
    queries = vectors[rng.choice(args.base_n, size=args.nq, replace=False)] + rng.normal(
        size=(args.nq, args.dim)
    ).astype(np.float32) * 0.05

    rows: List[Dict[str, float]] = []
    for ratio in args.delta_ratios:
        insert_n = max(1, int(args.base_n * ratio))
        new_ids = np.arange(10_000_000 + int(ratio * 1_000_000), 10_000_000 + int(ratio * 1_000_000) + insert_n)
        delta_assign = rng.integers(0, args.nlist, size=insert_n)
        delta_vecs = centroids[delta_assign] + rng.normal(size=(insert_n, args.dim)).astype(np.float32) * 0.6

        _, insert_ms = elapsed_ms(lambda: service.insert(new_ids, delta_vecs))

        if ratio >= args.delta_ivf_ratio:
            _, delta_ivf_ms = elapsed_ms(lambda: service.build_delta_ivf(args.delta_nlist, seed=17))
        else:
            delta_ivf_ms = 0.0

        _, exact_ms = elapsed_ms(lambda: service.search_exact(queries, k=10, nprobe=args.nprobe))
        _, pq_ms = elapsed_ms(lambda: service.search_pq(queries, k=10, nprobe=args.nprobe, rerank_budget=64))

        delete_ids = new_ids[: max(1, insert_n // 10)]
        _, delete_ms = elapsed_ms(lambda: service.delete(delete_ids))
        _, post_delete_ms = elapsed_ms(lambda: service.search_pq(queries, k=10, nprobe=args.nprobe, rerank_budget=64))

        needs = service.needs_compaction()
        _, compact_ms = elapsed_ms(lambda: service.compact(refresh_pq=False)) if needs else (None, 0.0)

        rows.append(
            {
                "delta_ratio": float(ratio),
                "insert_n": float(insert_n),
                "insert_ms": insert_ms,
                "delta_ivf_ms": delta_ivf_ms,
                "exact_search_ms": exact_ms,
                "pq_search_ms": pq_ms,
                "delete_ms": delete_ms,
                "pq_search_after_delete_ms": post_delete_ms,
                "needs_compaction": float(needs),
                "compact_ms": compact_ms,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-n", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--nq", type=int, default=128)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--pq-m", type=int, default=4)
    parser.add_argument("--delta-nlist", type=int, default=16)
    parser.add_argument("--delta-ivf-ratio", type=float, default=0.02)
    parser.add_argument("--delta-ratios", type=float, nargs="+", default=[0.001, 0.01, 0.05])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("dynamic_updates/update_benchmark.csv"))
    args = parser.parse_args()

    rows = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            "delta={delta_ratio:.3f} insert_ms={insert_ms:.2f} "
            "exact_ms={exact_search_ms:.2f} pq_ms={pq_search_ms:.2f} "
            "post_delete_pq_ms={pq_search_after_delete_ms:.2f} compact_ms={compact_ms:.2f}".format(**row)
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
