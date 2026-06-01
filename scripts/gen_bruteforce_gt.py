#!/usr/bin/env python3
"""
Brute-force exact top-K ground-truth generator for SIFT subsets.

Output format:
    int32 npts
    int32 k
    int32 idx[npts * k]

Example:
    python3 gen_bruteforce_gt.py /dev/shm/sift1b 10m 100 gpu
"""
import os
import struct
import sys
import time

import numpy as np


def read_diskann_u8bin(path, max_n=None):
    """DiskANN u8bin: [npts:int32][dim:int32][uint8 data]"""
    with open(path, "rb") as f:
        hdr = f.read(8)
        npts, dim = struct.unpack("ii", hdr)
        file_sz = os.path.getsize(path)
        avail = (file_sz - 8) // dim
        n = min(npts, avail)
        if max_n is not None and max_n > 0:
            n = min(n, max_n)
        raw = np.frombuffer(f.read(n * dim), dtype=np.uint8).reshape(n, dim)
    return raw.astype(np.float32), n, dim


def write_diskann_gt(path, idx):
    """ DiskANN gt.bin: [npts:i32][k:i32][i32 idx[npts*k]]"""
    n, k = idx.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", int(n), int(k)))
        idx.astype(np.int32).tofile(f)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    data_dir = sys.argv[1]
    scale    = sys.argv[2]                    # 10m | 100m
    k        = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    backend  = sys.argv[4] if len(sys.argv) >= 5 else "auto"  # auto|gpu|cpu

    def parse_scale_n(sx: str) -> int:
        unit = sx[-1].lower()
        v = float(sx[:-1])
        return int(v * (1e9 if unit == "b" else 1e6))

    def strip_deep(s: str) -> str:
        return s[4:] if s.startswith("deep") else s

    short = strip_deep(scale)
    try:
        cap = parse_scale_n(short)
    except Exception:
        print(f"[FATAL] invalid scale={scale}"); sys.exit(1)

    base_path  = os.path.join(data_dir, f"base_{short}.bin")
    query_path = os.path.join(data_dir, "query.bin")
    gt_path    = os.path.join(data_dir, f"groundtruth_{scale}.bin")

    print(f"======== brute-force GT generator ({scale.upper()}) ========")
    print(f"  data_dir  = {data_dir}")
    print(f"  base      = {base_path}")
    print(f"  query     = {query_path}")
    print(f"  k         = {k}")
    print(f"  backend   = {backend}")
    print(f"  out       = {gt_path}")
    print("=" * 56)

    t0 = time.time()
    base, n_base, dim = read_diskann_u8bin(base_path, max_n=cap)
    print(f"[IO] base  shape={base.shape}  {time.time()-t0:.1f}s  "
          f"mem={base.nbytes/1024**3:.1f} GB (fp32)")

    t0 = time.time()
    query, n_query, dim_q = read_diskann_u8bin(query_path, max_n=None)
    print(f"[IO] query shape={query.shape} {time.time()-t0:.1f}s")
    assert dim == dim_q, f"base dim {dim} != query dim {dim_q}"

    #  backend
    import faiss
    print(f"[FAISS] version = {faiss.__version__}")
    try_gpu = backend in ("auto", "gpu")
    gpu_ok = False
    if try_gpu:
        try:
            res = faiss.StandardGpuResources()
            #  co_cpu_to_gpu_list  index_cpu_to_gpu
            cpu_index = faiss.IndexFlatL2(dim)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_ok = True
            print("[FAISS] GPU path available")
        except Exception as e:
            print(f"[FAISS] GPU unavailable ({e}); fall back to CPU")

    if gpu_ok:
        # GPU  10M  128  4B = 5.1 GB100M = 51 GB
        base_gb = base.nbytes / 1024**3
        if base_gb > 7.0:
            print(f"[FAISS] base {base_gb:.1f} GB > 7 GB, GPU  CPU")
            gpu_ok = False

    if gpu_ok:
        t0 = time.time()
        gpu_index.add(np.ascontiguousarray(base))
        print(f"[GPU] add {time.time()-t0:.1f}s")
        t0 = time.time()
        #  base + 10k
        chunk = 1024
        I_all = np.empty((n_query, k), dtype=np.int64)
        for i in range(0, n_query, chunk):
            _D, I = gpu_index.search(query[i:i+chunk], k)
            I_all[i:i+chunk] = I
        print(f"[GPU] search {time.time()-t0:.1f}s")
        idx = I_all.astype(np.int32)
    else:
        # CPU:
        faiss.omp_set_num_threads(os.cpu_count() or 64)
        cpu_index = faiss.IndexFlatL2(dim)
        t0 = time.time()
        cpu_index.add(np.ascontiguousarray(base))
        print(f"[CPU] add {time.time()-t0:.1f}s  nthreads={faiss.omp_get_max_threads()}")
        t0 = time.time()
        chunk = 64  # 10M  128  64  0.3 GB
        I_all = np.empty((n_query, k), dtype=np.int64)
        for i in range(0, n_query, chunk):
            _D, I = cpu_index.search(query[i:i+chunk], k)
            I_all[i:i+chunk] = I
            if (i // chunk) % 16 == 0:
                dt = time.time() - t0
                eta = dt / max(i + chunk, 1) * n_query
                print(f"[CPU] search {i+chunk:>6}/{n_query}  "
                      f"elapsed={dt:.1f}s  ETA={eta:.1f}s")
        print(f"[CPU] search total {time.time()-t0:.1f}s")
        idx = I_all.astype(np.int32)

    # sanity
    assert idx.min() >= 0 and idx.max() < n_base
    write_diskann_gt(gt_path, idx)
    print(f"[OUT] wrote {gt_path}  npts={n_query} k={k}  "
          f"size={os.path.getsize(gt_path)/1024**2:.1f} MB")
    print("[DONE]")


if __name__ == "__main__":
    main()
