#!/usr/bin/env python3
"""
BIGANN .bvecs  DiskANN u8bin

.bvecs BIGANN SIFT
  [4 B dim_le] [dim  1 B uint8 data]

DiskANN u8bin
  8 B header: [4 B n_le] [4 B dim_le]
  n  dim  1 B uint8 data

 4 B dim  4 B
"""
import sys
import os
import struct
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--expected-n", type=int, default=0,
                    help=" n0 = ")
    ap.add_argument("--chunk", type=int, default=10_000_000,
                    help="")
    args = ap.parse_args()

    src_size = os.path.getsize(args.src)
    with open(args.src, "rb") as fi:
        #  dim
        header = fi.read(4)
        if len(header) < 4:
            print(f"FATAL: {args.src} too small"); sys.exit(1)
        dim = struct.unpack("<i", header)[0]
        rec_bytes = 4 + dim
        n = src_size // rec_bytes
        if args.expected_n and n != args.expected_n:
            print(f"[WARN] detected n={n}, expected={args.expected_n}")
        print(f"[info] src={args.src}  dim={dim}  n={n}  size={src_size/1e9:.2f} GB")
        fi.seek(0)

        with open(args.dst, "wb") as fo:
            fo.write(struct.pack("<ii", n, dim))
            buf_size = args.chunk * rec_bytes
            written = 0
            processed = 0
            while True:
                raw = fi.read(buf_size)
                if not raw:
                    break
                mv = memoryview(raw)
                n_in_chunk = len(raw) // rec_bytes
                for i in range(n_in_chunk):
                    off = i * rec_bytes + 4      # skip 4-byte dim header per vec
                    fo.write(mv[off:off + dim])
                processed += n_in_chunk
                written += n_in_chunk * dim
                print(f"  progress {processed}/{n}  ({100*processed/n:.1f}%)  "
                      f"written {written/1e9:.2f} GB", flush=True)

    out_size = os.path.getsize(args.dst)
    expect = 8 + n * dim
    assert out_size == expect, f"out_size={out_size} expect={expect}"
    print(f"[OK] wrote {args.dst}  {out_size/1e9:.2f} GB  (n={n}, dim={dim})")


if __name__ == "__main__":
    main()
