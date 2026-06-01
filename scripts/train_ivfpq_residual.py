#!/usr/bin/env python3
"""
Offline training + encoding for Residual-PQ + rerank.


  - base DiskANN u8bin  fbin: [n:i32][dim:i32][payload]
  - centroids  dump [nlist:i32][dim:i32][fp32 nlist*dim]
  - assign  dump [n:i64][i32 n]assign[v]  v  base  cluster id

dump
  - codebook
      [M:i32][K:i32][d_sub:i32][pad:i32][float32 M*K*d_sub]
  - pq_codes ** cluster-reorder ** h_base_u8
      [n:i64][M:i32][pad:i32][uint8 n*M]


  1.  centroids + assign
  2.  reorder_index[reorder_pos]  original_base_id
      C++  dataset.reordered_indices
  3.  train_N  residual  PQ faiss.ProductQuantizer
  4.  cluster  base residual codebook    pq_codes
"""
import argparse
import json
import os
import resource
import struct
import sys
import time
import numpy as np


VARIABLE_CODEBOOK_FORMAT = 1


class VariablePQ:
    def __init__(self, subdims, suboffsets, codebooks, indices):
        self.subdims = subdims
        self.suboffsets = suboffsets
        self.codebooks = codebooks
        self.indices = indices

    def compute_codes(self, resid):
        codes = np.empty((resid.shape[0], len(self.subdims)), dtype=np.uint8)
        for m, (dsub, off, index) in enumerate(zip(self.subdims, self.suboffsets, self.indices)):
            _, ids = index.search(np.ascontiguousarray(resid[:, off:off + dsub], dtype=np.float32), 1)
            codes[:, m] = ids[:, 0].astype(np.uint8)
        return codes


def make_subdims(dim, M):
    base = dim // M
    rem = dim % M
    if base <= 0:
        raise ValueError(f"M={M} must be <= dim={dim}")
    return np.array([base + 1 if i < rem else base for i in range(M)], dtype=np.int32)


def make_suboffsets(subdims):
    out = np.zeros(len(subdims), dtype=np.int32)
    if len(subdims) > 1:
        out[1:] = np.cumsum(subdims[:-1], dtype=np.int32)
    return out


def read_diskann_bin_header(path):
    with open(path, "rb") as f:
        npts, dim = struct.unpack("ii", f.read(8))
    return npts, dim


def infer_base_format(path, requested="auto"):
    if requested != "auto":
        return requested
    return "fbin" if path.endswith(".fbin") else "u8bin"


def read_centroids(path):
    """Our dump: [nlist:i32][dim:i32][float32 nlist*dim]"""
    with open(path, "rb") as f:
        nl, d = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(nl * d * 4), dtype=np.float32).reshape(nl, d).copy()
    return data  # [nlist, dim] float32


def read_assign(path):
    """Our dump: [n:i64][i32 n]"""
    with open(path, "rb") as f:
        n = struct.unpack("q", f.read(8))[0]
        a = np.frombuffer(f.read(n * 4), dtype=np.int32).copy()
    return a  # [n] int32


def read_base_rows(path, row_indices, dim, cap=None, chunk=2_000_000, base_format="u8bin"):
    """ index  base

     raw file
     /dev/shm  NVMe
    """
    dtype = np.float32 if base_format == "fbin" else np.uint8
    elem_size = np.dtype(dtype).itemsize
    row_indices = np.asarray(row_indices, dtype=np.int64)
    order = np.argsort(row_indices)
    sorted_idx = row_indices[order]
    out = np.zeros((row_indices.size, dim), dtype=np.float32)

    with open(path, "rb") as f:
        hdr = f.read(8)
        n_header, d_header = struct.unpack("ii", hdr)
        if d_header != dim:
            raise ValueError(f"dim mismatch: file dim={d_header}, expected {dim}")
        nmax = n_header if cap is None else min(cap, n_header)

        scan_pos = 0
        read_ptr = 0  #
        while scan_pos < sorted_idx.size:
            target = int(sorted_idx[scan_pos])
            if target >= nmax:
                break
            # seek
            if target > read_ptr:
                f.seek(8 + target * dim * elem_size)
                read_ptr = target
            #
            start = target
            end = min(nmax, start + chunk)
            buf = np.frombuffer(f.read((end - start) * dim * elem_size),
                                dtype=dtype).reshape(end - start, dim)
            read_ptr = end
            #  [start, end)
            while scan_pos < sorted_idx.size and sorted_idx[scan_pos] < end:
                local = int(sorted_idx[scan_pos]) - start
                out[order[scan_pos]] = buf[local].astype(np.float32)
                scan_pos += 1
    return out  # [N, dim] float32


def build_reorder_order(assign, nlist):
    """ base  cluster id cluster-reorder  original base id

     reorder_to_original: [n] int32
    reorder_to_original[reorder_pos] = original_id
     dataset.reordered_indices  cluster id  cluster  original
    """
    n = assign.shape[0]
    #  np.argsort(kind='stable')  cluster
    order = np.argsort(assign, kind="stable").astype(np.int32)
    return order  # [n] int32


def train_pq_residual(
    base_path, centroids, assign,
    train_n, M, K, out_codebook_path, n_base=None, seed=42, base_format="u8bin",
    variable_subdims=False,
):
    """ train_n  base   residual  faiss ProductQuantizer.train

    n_base:  base  = assign.shape[0] None
    prefix  header  1B
    """
    import faiss
    n_hdr, dim = read_diskann_bin_header(base_path)
    n = int(n_base) if n_base is not None else n_hdr
    if not variable_subdims:
        assert dim % M == 0, f"dim={dim} not divisible by M={M}; use --variable-subdims"
        d_sub = dim // M
    else:
        d_sub = -1

    rng = np.random.default_rng(seed)
    train_n = int(min(train_n, n))
    sample_ids = rng.choice(n, size=train_n, replace=False)

    print(f"[TRAIN] sampling {train_n}/{n} rows from base "
          f"(header says {n_hdr}) ...", flush=True)
    t0 = time.time()
    sampled = read_base_rows(base_path, sample_ids, dim, cap=n, base_format=base_format)
    print(f"[TRAIN]   sample read: {time.time()-t0:.1f}s", flush=True)

    # residual = base - centroid[assign[v]]
    resid = sampled - centroids[assign[sample_ids]]
    print(f"[TRAIN]   residual stats: mean={resid.mean():+.3f} "
          f"std={resid.std():.3f} (vs sampled std={sampled.std():.3f})", flush=True)

    if variable_subdims:
        subdims = make_subdims(dim, M)
        suboffsets = make_suboffsets(subdims)
        print(f"[TRAIN] variable PQ train M={M} K={K} subdims={subdims.tolist()} ...", flush=True)
        t0 = time.time()
        codebooks = []
        indices = []
        import faiss
        for m, (dsm, off) in enumerate(zip(subdims, suboffsets)):
            x = np.ascontiguousarray(resid[:, off:off + dsm], dtype=np.float32)
            km = faiss.Kmeans(int(dsm), K, niter=25, verbose=False, seed=seed + m)
            km.train(x)
            cbm = np.ascontiguousarray(km.centroids.reshape(K, int(dsm)), dtype=np.float32)
            idx = faiss.IndexFlatL2(int(dsm))
            idx.add(cbm)
            codebooks.append(cbm)
            indices.append(idx)
            print(f"[TRAIN]   subspace {m+1}/{M} d={int(dsm)} trained", flush=True)
        print(f"[TRAIN]   variable PQ train: {time.time()-t0:.1f}s", flush=True)
        cb_offsets = np.zeros(M, dtype=np.int64)
        acc = 0
        for m, dsm in enumerate(subdims):
            cb_offsets[m] = acc
            acc += K * int(dsm)
        with open(out_codebook_path, "wb") as f:
            f.write(struct.pack("iiii", M, K, -1, VARIABLE_CODEBOOK_FORMAT))
            f.write(subdims.astype(np.int32).tobytes())
            f.write(suboffsets.astype(np.int32).tobytes())
            f.write(cb_offsets.astype(np.int64).tobytes())
            for cbm in codebooks:
                f.write(cbm.tobytes())
        cb_bytes = sum(c.nbytes for c in codebooks)
        print(f"[TRAIN] variable codebook -> {out_codebook_path}  ({cb_bytes/1024:.1f} KB)",
              flush=True)
        return VariablePQ(subdims, suboffsets, codebooks, indices), codebooks

    print(f"[TRAIN] faiss PQ train M={M} K={K} d_sub={d_sub} ...", flush=True)
    t0 = time.time()
    pq = faiss.ProductQuantizer(dim, M, int(np.log2(K)))
    pq.train(resid.astype("float32"))
    print(f"[TRAIN]   PQ.train: {time.time()-t0:.1f}s", flush=True)

    # codebook: faiss stores centroids as [M*K, d_sub] float32
    cb = faiss.vector_to_array(pq.centroids).reshape(M, K, d_sub).astype("float32")

    # write
    with open(out_codebook_path, "wb") as f:
        f.write(struct.pack("iiii", M, K, d_sub, 0))
        f.write(cb.tobytes())
    print(f"[TRAIN] codebook -> {out_codebook_path}  ({cb.nbytes/1024:.1f} KB)",
          flush=True)
    return pq, cb


def encode_all(
    base_path, centroids, assign,
    reorder_to_original,
    pq,  # faiss.ProductQuantizer
    M, K,
    out_codes_path,
    n_base=None,
    chunk=4_000_000,
    base_format="u8bin",
):
    """ 1B  cluster-reorder  PQ codes


       reorder_pos = 0..n-1
      original_id = reorder_to_original[reorder_pos]
      base_v = base.row(original_id)
      residual = base_v - centroids[assign[original_id]]
      code = pq.compute_codes([residual])[0]
       code  reorder_pos*M

     1B    chunk  raw base
     original_id  code scatter

     outer_chunkoriginal  scatter
     pq_codes_reordered[n, M]** n*M bytes **1B16=16GB
     pass
    """
    dtype = np.float32 if base_format == "fbin" else np.uint8
    elem_size = np.dtype(dtype).itemsize
    n_hdr, dim = read_diskann_bin_header(base_path)
    n = int(n_base) if n_base is not None else n_hdr
    #  scatter original_id -> reorder_pos
    reorder_pos = np.empty_like(reorder_to_original)
    reorder_pos[reorder_to_original] = np.arange(n, dtype=np.int32)

    #  reorder
    pq_codes = np.empty((n, M), dtype=np.uint8)
    print(f"[ENC] allocating pq_codes: {pq_codes.nbytes/(1024**3):.2f} GB",
          flush=True)

    t0 = time.time()
    with open(base_path, "rb") as f:
        f.read(8)  # skip header
        pos = 0
        while pos < n:
            end = min(n, pos + chunk)
            cnt = end - pos
            raw = np.frombuffer(f.read(cnt * dim * elem_size), dtype=dtype).reshape(cnt, dim)
            # residual
            centers = centroids[assign[pos:end]]          # [cnt, dim] float32
            resid = raw.astype(np.float32) - centers      # [cnt, dim]
            # PQ encode
            codes = pq.compute_codes(resid)               # [cnt, M] uint8
            # scatter  reorder
            pq_codes[reorder_pos[pos:end]] = codes
            pos = end
            if (pos // chunk) % 5 == 0:
                elapsed = time.time() - t0
                rate = pos / max(elapsed, 1e-9)
                eta = (n - pos) / max(rate, 1e-9)
                print(f"[ENC]   {pos}/{n} ({100*pos/n:.1f}%)  "
                      f"rate={rate/1e6:.1f}M/s  eta={eta/60:.1f}min",
                      flush=True)

    # write
    with open(out_codes_path, "wb") as f:
        f.write(struct.pack("q", n))
        f.write(struct.pack("ii", M, 0))
        f.write(pq_codes.tobytes())
    print(f"[ENC] pq_codes -> {out_codes_path}  "
          f"({pq_codes.nbytes/(1024**3):.2f} GB)", flush=True)
    return time.time() - t0


def peak_rss_gb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024.0 / 1024.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",       required=True, help="DiskANN u8bin or fbin base file")
    p.add_argument("--base-format", choices=["auto", "u8bin", "fbin"], default="auto",
                   help="base vector format; auto infers .fbin as fp32")
    p.add_argument("--centroids",  required=True, help="centroids dump (from ivftensor)")
    p.add_argument("--assign",     required=True, help="assign dump (from ivftensor)")
    p.add_argument("--out-codebook", required=True)
    p.add_argument("--out-codes",    required=True)
    p.add_argument("--M", type=int, default=16, help="# subspaces")
    p.add_argument("--K", type=int, default=256, help="codes per subspace (must be power of 2)")
    p.add_argument("--variable-subdims", action="store_true",
                   help="Allow M that does not divide dim by training one KMeans per variable-width subspace.")
    p.add_argument("--train-n", type=int, default=1_000_000)
    p.add_argument("--chunk",   type=int, default=4_000_000, help="encode stream chunk")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--metrics-out", default="", help="optional JSON metrics path")
    args = p.parse_args()
    total_t0 = time.time()

    if args.K & (args.K - 1) != 0:
        raise SystemExit(f"K must be power of 2, got {args.K}")
    base_format = infer_base_format(args.base, args.base_format)
    print(f"[IO] base_format={base_format} base={args.base}", flush=True)

    print(f"[IO] centroids: {args.centroids}", flush=True)
    centroids = read_centroids(args.centroids)
    nlist, dim = centroids.shape
    print(f"     nlist={nlist} dim={dim}", flush=True)

    print(f"[IO] assign: {args.assign}", flush=True)
    assign = read_assign(args.assign)
    n = assign.shape[0]
    print(f"     n={n}", flush=True)

    # build reorder order (sorted by cluster id, stable)
    print(f"[REORDER] argsort(stable) n={n} ...", flush=True)
    t0 = time.time()
    reorder_to_original = build_reorder_order(assign, nlist)
    reorder_sec = time.time() - t0
    print(f"     done in {reorder_sec:.1f}s", flush=True)

    # train
    train_t0 = time.time()
    pq, cb = train_pq_residual(
        base_path=args.base,
        centroids=centroids,
        assign=assign,
        train_n=args.train_n,
        M=args.M, K=args.K,
        out_codebook_path=args.out_codebook,
        n_base=n,
        seed=args.seed,
        base_format=base_format,
        variable_subdims=args.variable_subdims,
    )
    train_sec = time.time() - train_t0

    # encode
    encode_sec = encode_all(
        base_path=args.base,
        centroids=centroids,
        assign=assign,
        reorder_to_original=reorder_to_original,
        pq=pq,
        M=args.M, K=args.K,
        out_codes_path=args.out_codes,
        n_base=n,
        chunk=args.chunk,
        base_format=base_format,
    )

    if args.metrics_out:
        metrics = {
            "base": args.base,
            "base_format": base_format,
            "centroids": args.centroids,
            "assign": args.assign,
            "out_codebook": args.out_codebook,
            "out_codes": args.out_codes,
            "n": int(n),
            "dim": int(dim),
            "nlist": int(nlist),
            "M": int(args.M),
            "K": int(args.K),
            "variable_subdims": bool(args.variable_subdims),
            "train_n": int(args.train_n),
            "chunk": int(args.chunk),
            "reorder_sec": float(reorder_sec),
            "train_plus_sample_sec": float(train_sec),
            "encode_sec": float(encode_sec),
            "total_sec": float(time.time() - total_t0),
            "codebook_bytes": os.path.getsize(args.out_codebook) if os.path.exists(args.out_codebook) else 0,
            "codes_bytes": os.path.getsize(args.out_codes) if os.path.exists(args.out_codes) else 0,
            "peak_rss_gb": peak_rss_gb(),
        }
        tmp = args.metrics_out + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, args.metrics_out)
        print(f"[METRICS] -> {args.metrics_out}", flush=True)


if __name__ == "__main__":
    main()
