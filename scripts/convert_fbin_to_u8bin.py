#!/usr/bin/env python3
"""
DiskANN fbin (fp32)  DiskANN u8bin

 DEEP1B  fp32 [-0.1, 0.1]unit-normalized
 MinMax scale  [0,255]  uint8 u8bin

DEEP-1B  L2 ANN
 fp32  reader  diskann_fp32_bin  uint8
 **** fp32 C++ reader  fp32 bin
"""
import argparse
import os
import struct
import sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--chunk", type=int, default=10_000_000,
                    help="")
    ap.add_argument("--scale-min", type=float, default=None,
                    help=" min")
    ap.add_argument("--scale-max", type=float, default=None,
                    help=" max")
    args = ap.parse_args()

    with open(args.src, "rb") as fi:
        hdr = fi.read(8)
        n, dim = struct.unpack("<ii", hdr)
        print(f"[info] {args.src}  n={n}  dim={dim}")

        if args.scale_min is None or args.scale_max is None:
            print("[info] scanning for min/max (sample 1M vectors)...")
            fi.seek(8)
            sample_n = min(1_000_000, n)
            raw = np.frombuffer(fi.read(sample_n * dim * 4),
                                dtype=np.float32).reshape(sample_n, dim)
            mn, mx = float(raw.min()), float(raw.max())
            del raw
            print(f"[info] sample min={mn:.6f} max={mx:.6f}")
        else:
            mn, mx = args.scale_min, args.scale_max

        # quantize to [0, 255]
        scale = 255.0 / (mx - mn) if mx > mn else 1.0

        fi.seek(8)
        with open(args.dst, "wb") as fo:
            fo.write(struct.pack("<ii", n, dim))
            processed = 0
            while processed < n:
                take = min(args.chunk, n - processed)
                raw = np.frombuffer(fi.read(take * dim * 4),
                                    dtype=np.float32).reshape(take, dim)
                q = np.clip((raw - mn) * scale, 0, 255).astype(np.uint8)
                fo.write(q.tobytes())
                processed += take
                print(f"  {processed}/{n}  ({100*processed/n:.1f}%)",
                      flush=True)

    out_size = os.path.getsize(args.dst)
    expect = 8 + n * dim
    assert out_size == expect
    #
    with open(args.dst + ".scale", "w") as fs:
        fs.write(f"{mn}\n{mx}\n{scale}\n")
    print(f"[OK] wrote {args.dst}  {out_size/1e9:.2f} GB")
    print(f"     scale meta -> {args.dst}.scale")


if __name__ == "__main__":
    main()
