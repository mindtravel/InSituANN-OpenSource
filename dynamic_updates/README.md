# Snapshot-Delta Update Prototype

This directory is an isolated prototype for online insert/delete support.  It
does not modify the existing CUDA search runners.

## Model

The serving state is:

```text
ActiveState = immutable main IVF snapshot
            + published delta index
            + tombstone bitmap
```

Queries pin the current `ActiveState` at the beginning of a request and use it
for the whole search.  Inserts are written to a staging delta and published by
an atomic state swap.  Deletes publish a new tombstone bitmap.  Rebuild /
compaction builds a private new snapshot, folds in live delta vectors, removes
tombstoned vectors, and swaps it in only after the new snapshot is complete.

Search is split into the intended two-stage pipeline boundary:

```text
Stage 1, GPU-side metadata stage:
    main IVF routing -> top-nprobe centroid ids
    delta index search -> top delta candidates

Stage 2, host-side verification stage:
    scan main IVF lists selected by Stage 1
    merge with Stage-1 delta candidates
    exact rerank over host-resident full vectors
```

## Implemented semantics

- Main snapshot is never modified in place.
- Inserted vectors are stored as full vectors in the delta store.
- PQ-enabled inserts are encoded with the current residual-PQ codebook.
- Exact search scans main IVF lists plus the delta index.
- PQ search scans main/delta compact codes, skips tombstones before local
  candidate selection, and runs final exact rerank over retained full vectors.
- `route_and_delta_stage()` models the fused GPU-side stage that returns both
  main top-nprobe centroids and bounded delta top candidates to the host.
- Delta can be flat or converted into a small delta-IVF.
- Compaction clears delta and tombstones after building a new immutable
  snapshot.

## Synthetic benchmark

Run:

```powershell
python -m dynamic_updates.benchmark_updates --out dynamic_updates/update_benchmark.csv
```

This benchmark uses small synthetic data to check the update layer.  It is
not a paper performance number.  Production performance must be measured with
the CUDA runners after this semantic layer is integrated.

## Real dataset sample benchmark

The remote benchmark script validates the reference update layer on real SIFT1B
and DEEP1B files while keeping the sample size small enough for NumPy:

```bash
python3 -m dynamic_updates.benchmark_real_datasets \
  --datasets sift1b deep1b \
  --base-n 20000 \
  --query-n 128 \
  --nlist 256 \
  --nprobe 16 \
  --pq-train-n 8000 \
  --ksub 16 \
  --pq-iters 4 \
  --delta-ratios 0.001 0.01 0.05 \
  --max-insert-n 2000 \
  --delta-nlist 64 \
  --out dynamic_updates/real_dataset_update_benchmark.csv
```

This uses real base/query vectors and production centroid files, but only a
prefix sample and a centroid prefix.  The output is a correctness and overhead
sanity check for the update layer, not a replacement for full CUDA throughput
experiments.

## CUDA runner status

The full CUDA update-performance runners are intentionally excluded from this
code release.  This directory documents the update semantics and keeps
the small reference implementation used to validate snapshot pinning, delta
publication, tombstone filtering, and compaction behavior.
