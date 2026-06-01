#!/usr/bin/env python3
"""Lightweight consistency checks for the public release artifact."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PQ_UPDATE = (ROOT / "updates" / "pq_update.cu").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
SIFT100M_BLOCK = PQ_UPDATE.split('} else if(ds=="sift100m"){', 1)[1].split('} else if(ds=="sift1b"){', 1)[0]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require("`n#include" not in PQ_UPDATE, "pq_update.cu contains a literal PowerShell newline token")
    require("nlist=131072" in SIFT100M_BLOCK and 'root="/workspace/results/sift100m"' in SIFT100M_BLOCK,
            "sift100m runner must use the paper nlist=131072")
    require("pq_m=32; pq_dsub=4;" in SIFT100M_BLOCK,
            "sift100m runner must use the paper M=32 residual-PQ layout")
    require("pq_highm_full100m_iter10/codebook_resid_M32_100m_nlist131072_full100mKMeans_iter10_train4m.bin" in SIFT100M_BLOCK,
            "sift100m runner must point to the M32/nlist131072 codebook")
    require("const int max_delta_n=100000000;" in PQ_UPDATE,
            "PQ update runner must allocate/build the 100M delta segment")
    require("{100000000,100000000}" in PQ_UPDATE,
            "PQ update runner must include the 100M/100M update pair")
    require("PQ-GPU update release currently covers query-time overlay execution" in README,
            "README must scope PQ update to the query-time overlay implementation")


if __name__ == "__main__":
    main()
