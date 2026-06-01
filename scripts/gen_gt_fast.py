#!/usr/bin/env python3
"""
Fast brute-force exact top-K groundtruth generator.

vs. gen_bruteforce_gt.py:
  -  chunk1024 queries / chunk Python overhead
  - CPU  faiss.omp_set_num_threads + IndexFlatL2
  -  scale1m / 10m / 40m / 100m / 1b / deep10m / deep50m / deep1b
  -  faiss GPU

Usage:
  python3 scripts/gen_gt_fast.py <data_dir> <scale> [k=100]
"""
import os
import struct
import sys
import time

import numpy as np


def read_u8bin(path, max_n=None):
    with open(path, "rb") as f:
        npts, dim = struct.unpack("ii", f.read(8))
        file_sz = os.path.getsize(path)
        avail = (file_sz - 8) // dim
        n = min(npts, avail)
        if max_n is not None and max_n > 0:
            n = min(n, max_n)
        raw = np.frombuffer(f.read(n * dim), dtype=np.uint8).reshape(n, dim)
    return raw, n, dim


def write_gt(path, idx):
    n, k = idx.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", int(n), int(k)))
        idx.astype(np.int32).tofile(f)


def parse_scale_n(sx):
    unit = sx[-1].lower()
    v = float(sx[:-1])
    return int(v * (1e9 if unit == "b" else 1e6))


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    data_dir = sys.argv[1]
    scale    = sys.argv[2]
    k        = int(sys.argv[3]) if len(sys.argv) >= 4 else 100

    short = scale[4:] if scale.startswith("deep") else scale
    cap   = parse_scale_n(short)

    base_path  = os.path.join(data_dir, f"base_{short}.bin")
    query_path = os.path.join(data_dir, "query.bin")
    gt_path    = os.path.join(data_dir, f"groundtruth_{scale}.bin")

    print(f"[GT] scale={scale} cap={cap:,} k={k}")
    print(f"     base={base_path}")
    print(f"     query={query_path}")
    print(f"     out={gt_path}")

    t0 = time.time()
    base_u8, n_base, dim = read_u8bin(base_path, max_n=cap)
    print(f"[IO] base {base_u8.shape} {time.time()-t0:.1f}s")
    t0 = time.time()
    query_u8, n_query, dim_q = read_u8bin(query_path, max_n=None)
    print(f"[IO] query {query_u8.shape} {time.time()-t0:.1f}s")
    assert dim == dim_q

    t0 = time.time()
    base_f32  = base_u8.astype(np.float32)
    query_f32 = query_u8.astype(np.float32)
    print(f"[CVT] u8->f32 base={base_f32.nbytes/2**30:.2f}GB "
          f"query={query_f32.nbytes/2**20:.1f}MB {time.time()-t0:.1f}s")

    import faiss
    try:
        nthreads = os.cpu_count() or 64
        faiss.omp_set_num_threads(nthreads)
        print(f"[FAISS] version={faiss.__version__} nthreads={nthreads}")
    except Exception:
        pass

    #  GPU
    use_gpu = False
    gpu_res = None
    try:
        gpu_res = faiss.StandardGpuResources()
        use_gpu = True
        print("[FAISS] GPU path available")
    except Exception as e:
        print(f"[FAISS] GPU unavailable: {e}")

    # 3060 Ti 8GBfp32 base  ~6GB  > 12M  tile
    GPU_BUDGET_BYTES = int(6.0 * 2**30)
    if use_gpu:
        tile_n = max(1, GPU_BUDGET_BYTES // (dim * 4))
        n_tiles = (n_base + tile_n - 1) // tile_n
        print(f"[GPU] tile_n={tile_n:,} n_tiles={n_tiles}")

        #  tile  GPU  IndexFlatL2 queries top-k
        t0 = time.time()
        I_all_global = np.full((n_query, k), -1, dtype=np.int64)
        D_all_global = np.full((n_query, k), np.inf, dtype=np.float32)
        for t in range(n_tiles):
            t_start = t * tile_n
            t_end   = min(n_base, t_start + tile_n)
            sub = base_f32[t_start:t_end]
            cpu_idx = faiss.IndexFlatL2(dim)
            g_idx   = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_idx)
            g_idx.add(np.ascontiguousarray(sub))

            chunk = 512
            for i in range(0, n_query, chunk):
                qb = query_f32[i:i+chunk]
                D, I = g_idx.search(np.ascontiguousarray(qb), k)
                I = I + t_start    # local -> global idx
                # merge with previous top-k
                D_cur = D_all_global[i:i+chunk]
                I_cur = I_all_global[i:i+chunk]
                D_cat = np.concatenate([D_cur, D], axis=1)
                I_cat = np.concatenate([I_cur, I], axis=1)
                order = np.argpartition(D_cat, k, axis=1)[:, :k]
                rows  = np.arange(D_cat.shape[0])[:, None]
                D_sel = D_cat[rows, order]
                I_sel = I_cat[rows, order]
                # sort the top-k by distance ascending
                sort_ord = np.argsort(D_sel, axis=1)
                D_all_global[i:i+chunk] = D_sel[rows, sort_ord]
                I_all_global[i:i+chunk] = I_sel[rows, sort_ord]

            del g_idx, cpu_idx, sub
            dt = time.time() - t0
            print(f"[GPU] tile {t+1}/{n_tiles} done  elapsed={dt:.1f}s")

        I_all = I_all_global
        dt = time.time() - t0
        print(f"[GPU] search all {n_query} queries  {n_base} base in {dt:.1f}s "
              f"({n_query*n_base/dt/1e9:.1f} G dist/s)")
    else:
        index = faiss.IndexFlatL2(dim)
        t0 = time.time()
        index.add(np.ascontiguousarray(base_f32))
        print(f"[CPU] add {time.time()-t0:.1f}s")

        t0 = time.time()
        chunk = 2048
        I_all = np.empty((n_query, k), dtype=np.int64)
        for i in range(0, n_query, chunk):
            qb = query_f32[i:i+chunk]
            _D, I = index.search(np.ascontiguousarray(qb), k)
            I_all[i:i+chunk] = I
        dt = time.time() - t0
        print(f"[CPU] search all {n_query} queries in {dt:.1f}s "
              f"({n_query*n_base/dt/1e9:.1f} G dist/s)")

    assert I_all.min() >= 0 and I_all.max() < n_base
    write_gt(gt_path, I_all.astype(np.int32))
    print(f"[OUT] wrote {gt_path} size={os.path.getsize(gt_path)/2**20:.1f} MB")


if __name__ == "__main__":
    main()
