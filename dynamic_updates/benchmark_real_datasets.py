"""Run the snapshot-delta prototype on real SIFT1B/DEEP1B vector samples.

This is a reference-layer benchmark, not a production CUDA number.  It uses
real dataset vectors and production centroids, but samples a manageable prefix
so that the pure Python/NumPy prototype can validate update behavior.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from dynamic_updates import ResidualPQEncoder, SnapshotDeltaIndex, build_snapshot


DATASETS = {
    "sift1b": {
        "dim": 128,
        "dtype": "uint8",
        "base": "/workspace/sift1b/base_1b.bin",
        "query": "/workspace/sift1b/query.bin",
        "centroids": "/workspace/results/sift1b/centroids/centroids_1b_nlist524288_train1b_iter10_8gpu.bin",
        "pq_m": 8,
    },
    "deep1b": {
        "dim": 96,
        "dtype": "float32",
        "base": "/workspace/data/deep1b/base_1b.fbin",
        "query": "/workspace/data/deep1b/query.fbin",
        "centroids": "/workspace/results/deep1b_kmeans_train1b_iter10_8gpu_20260508_183800/deep1b/centroids/centroids_deep1b_nlist524288_train1b_iter10_8gpu.bin",
        "pq_m": 8,
    },
}


def elapsed_ms(fn):
    t0 = time.perf_counter()
    value = fn()
    t1 = time.perf_counter()
    return value, (t1 - t0) * 1000.0


def read_header(path: str) -> tuple[int, int]:
    header = np.fromfile(path, dtype=np.int32, count=2)
    if header.shape[0] != 2:
        raise ValueError(f"missing 8-byte header: {path}")
    return int(header[0]), int(header[1])


def mmap_matrix(path: str, dtype: str, rows: int, dim: int, offset_rows: int = 0) -> np.ndarray:
    dtype_np = np.dtype(dtype)
    return np.memmap(
        path,
        dtype=dtype_np,
        mode="r",
        offset=8 + offset_rows * dim * dtype_np.itemsize,
        shape=(rows, dim),
    )


def load_sample(path: str, dtype: str, rows: int, dim: int, offset_rows: int = 0) -> np.ndarray:
    return np.asarray(mmap_matrix(path, dtype, rows, dim, offset_rows), dtype=np.float32)


def run_dataset(args: argparse.Namespace, dataset: str) -> List[Dict[str, float | str]]:
    cfg = DATASETS[dataset]
    dim = int(cfg["dim"])
    base_rows, base_dim = read_header(str(cfg["base"]))
    query_rows, query_dim = read_header(str(cfg["query"]))
    if base_dim != dim or query_dim != dim:
        raise ValueError(f"dimension mismatch for {dataset}: base={base_dim}, query={query_dim}, expected={dim}")

    base_n = min(args.base_n, base_rows - args.max_insert_n)
    query_n = min(args.query_n, query_rows)
    base = load_sample(str(cfg["base"]), str(cfg["dtype"]), base_n, dim, 0)
    queries = load_sample(str(cfg["query"]), str(cfg["dtype"]), query_n, dim, 0)
    centroids = load_sample(str(cfg["centroids"]), "float32", args.nlist, dim, 0)
    ids = np.arange(base_n, dtype=np.int64)

    snapshot, build_ms = elapsed_ms(lambda: build_snapshot(ids, base, centroids))
    pq_m = int(cfg["pq_m"])
    pq_train_n = min(args.pq_train_n, base_n)
    _, pq_train_ms = elapsed_ms(
        lambda: ResidualPQEncoder.train(
            snapshot.vectors[:pq_train_n],
            snapshot.centroids,
            snapshot.assignments[:pq_train_n],
            m=pq_m,
            ksub=args.ksub,
            iters=args.pq_iters,
            seed=29,
        )
    )
    pq = ResidualPQEncoder.train(
        snapshot.vectors[:pq_train_n],
        snapshot.centroids,
        snapshot.assignments[:pq_train_n],
        m=pq_m,
        ksub=args.ksub,
        iters=args.pq_iters,
        seed=29,
    )
    snapshot = build_snapshot(ids, base, centroids, pq)

    rows: List[Dict[str, float | str]] = []
    for ratio in args.delta_ratios:
        service = SnapshotDeltaIndex(
            snapshot,
            pq_encoder=pq,
            delta_ratio_threshold=args.compact_delta_ratio,
            delete_ratio_threshold=args.compact_delete_ratio,
        )
        insert_n = min(max(1, int(base_n * ratio)), args.max_insert_n)
        insert_offset = base_n
        insert_vectors = load_sample(str(cfg["base"]), str(cfg["dtype"]), insert_n, dim, insert_offset)
        insert_ids = np.arange(10_000_000 + insert_offset, 10_000_000 + insert_offset + insert_n, dtype=np.int64)

        _, insert_ms = elapsed_ms(lambda: service.insert(insert_ids, insert_vectors))
        if ratio >= args.delta_ivf_ratio:
            _, delta_ivf_ms = elapsed_ms(lambda: service.build_delta_ivf(args.delta_nlist, seed=31))
        else:
            delta_ivf_ms = 0.0

        _, exact_ms = elapsed_ms(lambda: service.search_exact(queries, k=10, nprobe=args.nprobe))
        _, pq_ms = elapsed_ms(lambda: service.search_pq(queries, k=10, nprobe=args.nprobe, rerank_budget=args.rerank_budget))

        delete_n = max(1, insert_n // 10)
        _, delete_ms = elapsed_ms(lambda: service.delete(insert_ids[:delete_n]))
        _, post_delete_pq_ms = elapsed_ms(
            lambda: service.search_pq(queries, k=10, nprobe=args.nprobe, rerank_budget=args.rerank_budget)
        )

        needs = service.needs_compaction()
        _, compact_ms = elapsed_ms(lambda: service.compact(refresh_pq=False)) if needs else (None, 0.0)

        rows.append(
            {
                "dataset": dataset,
                "base_rows_file": float(base_rows),
                "base_n_sample": float(base_n),
                "query_n": float(query_n),
                "dim": float(dim),
                "nlist_sample": float(args.nlist),
                "nprobe": float(args.nprobe),
                "delta_ratio": float(ratio),
                "insert_n": float(insert_n),
                "build_snapshot_ms": build_ms,
                "pq_train_ms": pq_train_ms,
                "insert_ms": insert_ms,
                "delta_ivf_ms": delta_ivf_ms,
                "exact_search_ms": exact_ms,
                "pq_search_ms": pq_ms,
                "delete_n": float(delete_n),
                "delete_ms": delete_ms,
                "pq_search_after_delete_ms": post_delete_pq_ms,
                "needs_compaction": float(needs),
                "compact_ms": compact_ms,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["sift1b", "deep1b"])
    parser.add_argument("--base-n", type=int, default=20000)
    parser.add_argument("--query-n", type=int, default=128)
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--pq-train-n", type=int, default=8000)
    parser.add_argument("--ksub", type=int, default=16)
    parser.add_argument("--pq-iters", type=int, default=4)
    parser.add_argument("--delta-ratios", type=float, nargs="+", default=[0.001, 0.01, 0.05])
    parser.add_argument("--max-insert-n", type=int, default=2000)
    parser.add_argument("--delta-ivf-ratio", type=float, default=0.02)
    parser.add_argument("--delta-nlist", type=int, default=64)
    parser.add_argument("--rerank-budget", type=int, default=64)
    parser.add_argument("--compact-delta-ratio", type=float, default=0.05)
    parser.add_argument("--compact-delete-ratio", type=float, default=0.10)
    parser.add_argument("--out", type=Path, default=Path("dynamic_updates/real_dataset_update_benchmark.csv"))
    args = parser.parse_args()

    all_rows: List[Dict[str, float | str]] = []
    for dataset in args.datasets:
        rows = run_dataset(args, dataset)
        all_rows.extend(rows)
        for row in rows:
            print(
                "{dataset} delta={delta_ratio:.3f} insert_ms={insert_ms:.2f} "
                "exact_ms={exact_search_ms:.2f} pq_ms={pq_search_ms:.2f} "
                "post_delete_pq_ms={pq_search_after_delete_ms:.2f} compact_ms={compact_ms:.2f}".format(**row)
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
