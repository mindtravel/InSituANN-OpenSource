#!/usr/bin/env bash
# ============================================================================
#  SIFT-1B / DEEP-{10M,50M,100M,1B}
#
#
#   bash scripts/download_datasets.sh            #
#   bash scripts/download_datasets.sh sift1b     #  SIFT-1B base
#   bash scripts/download_datasets.sh deep10m    #  DEEP-10M
#   bash scripts/download_datasets.sh deep50m
#   bash scripts/download_datasets.sh deep100m
#   bash scripts/download_datasets.sh deep1b
#
#
#   SIFT-1B base : /dev/shm/sift1b/base_1b.bin        (~128 GB uint8)
#   DEEP query   : /workspace/data/deep1b/query.bin            (u8bin)
#   DEEP base_NM : /workspace/data/deep1b/base_<N>m.bin         (u8bin)
#   DEEP fbin    : /workspace/data/deep1b/base_<N>m.fbin        ( fp32)
#   DEEP gt      : /workspace/data/deep1b/groundtruth_<N>m.bin  (big-ann )
#
# DEEP
#   base (fp32, 96-d, 1B  4B = 384 GB):
#     https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
#   query (10k  96, fp32, 3.84 MB):
#     https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
#   groundtruth (Top-100 ids):
#     10M:  https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/deep-10M
#     100M: https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/deep-100M
#     1B:   https://storage.yandexcloud.net/yandex-research/ann-datasets/deep_new_groundtruth.public.10K.bin
#   DEEP-{10M,100M} base = base.1B.fbin  N  big-ann-benchmarks datasets.py
#
#
#   fbin : [n:uint32][dim:uint32][fp32  n  dim]
#   u8bin: [n:int32][dim:int32][uint8  n  dim]
#
#  aria2c
#   apt-get update && apt-get install -y aria2
# ============================================================================
set -euo pipefail

SIFT_DIR="${SIFT_DIR:-/dev/shm/sift1b}"
DEEP_DIR="${DEEP_DIR:-/workspace/data/deep1b}"
WHAT="${1:-all}"

# --- DEEP  Yandex CDN ---
DEEP_BASE_URL="${DEEP_BASE_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin}"
DEEP_QUERY_URL="${DEEP_QUERY_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin}"
DEEP_GT_10M_URL="${DEEP_GT_10M_URL:-https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M/deep-10M}"
DEEP_GT_100M_URL="${DEEP_GT_100M_URL:-https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/deep-100M}"
DEEP_GT_1B_URL="${DEEP_GT_1B_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/deep_new_groundtruth.public.10K.bin}"

# DEEP
DEEP_DIM="${DEEP_DIM:-96}"
DEEP_N_FULL="${DEEP_N_FULL:-1000000000}"

#
DL_PARTS="${DL_PARTS:-8}"
DL_RETRIES="${DL_RETRIES:-10}"

#  u8bin  yes u8bin readerv3-u8  .fbin
DEEP_MAKE_U8BIN="${DEEP_MAKE_U8BIN:-1}"
#  u8bin DEEP  PCA+L2 single-vector ~ 0.55
#  headroom  clipstep = 1.2/255  0.0047 std  4.6%
DEEP_SCALE_MIN="${DEEP_SCALE_MIN:--0.6}"
DEEP_SCALE_MAX="${DEEP_SCALE_MAX:-0.6}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

ensure_tools() {
    if ! command -v aria2c >/dev/null 2>&1; then
        log " aria2c ..."
        apt-get update -y >/dev/null && apt-get install -y aria2 >/dev/null
    fi
    if ! command -v curl >/dev/null 2>&1; then
        apt-get update -y >/dev/null && apt-get install -y curl >/dev/null
    fi
}

# ----------------------------------------------------------------------------
# scale_to_n  '10m' -> 10_000_000, '50m' -> 50_000_000, '1b' -> 1_000_000_000
# ----------------------------------------------------------------------------
scale_to_n() {
    local s="${1,,}"
    case "$s" in
        *b) echo $(( ${s%b} * 1000000000 )) ;;
        *m) echo $(( ${s%m} * 1000000 )) ;;
        *)  echo "unknown scale: $s" >&2; return 1 ;;
    esac
}

# ----------------------------------------------------------------------------
# parallel_range_download URL OUT TOTAL_BYTES [PARTS]
#
#  curl  URL  TOTAL_BYTES Yandex CDN  HTTP Range
#  == TOTAL_BYTES Range
# ----------------------------------------------------------------------------
parallel_range_download() {
    local url="$1"
    local out="$2"
    local total="$3"
    local parts="${4:-$DL_PARTS}"
    local chunk=$(( (total + parts - 1) / parts ))
    local tmpdir
    tmpdir="$(mktemp -d "${out}.tmpXXXX")"

    log "[DL] $(basename "$out") parts=${parts} chunk=${chunk} total=${total}"
    local pids=() i=0 start end
    for ((i=0; i<parts; i++)); do
        start=$(( i * chunk ))
        end=$(( start + chunk - 1 ))
        if (( end >= total )); then end=$(( total - 1 )); fi
        if (( start > end )); then continue; fi
        (
            curl -fSL --retry "$DL_RETRIES" --retry-delay 2 --retry-all-errors \
                 --range "${start}-${end}" \
                 -o "${tmpdir}/part.$(printf '%04d' "$i")" \
                 "$url" \
                 >/dev/null 2>&1 \
             || {
                 log "[DL][WARN] part $i failed, second try with verbose ..."
                 curl -fSL --retry "$DL_RETRIES" --retry-delay 5 --retry-all-errors \
                      --range "${start}-${end}" \
                      -o "${tmpdir}/part.$(printf '%04d' "$i")" \
                      "$url"
             }
        ) &
        pids+=($!)
    done
    local fail=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then fail=1; fi
    done
    if (( fail )); then
        log "[DL][FATAL] one or more parts failed for $url"
        rm -rf "$tmpdir"
        return 1
    fi

    cat "${tmpdir}"/part.* > "$out"
    rm -rf "$tmpdir"

    local got
    got="$(stat -c %s "$out")"
    if [ "$got" != "$total" ]; then
        log "[DL][FATAL] size mismatch: got=$got expect=$total ($out)"
        return 1
    fi
    log "[DL] done $(basename "$out")  bytes=${got}"
}

# ----------------------------------------------------------------------------
# rewrite_fbin_header file  n  dim
#  fbin  8  header  (n:uint32, dim:uint32)
# ----------------------------------------------------------------------------
rewrite_fbin_header() {
    local f="$1" n="$2" dim="$3"
    python3 - "$f" "$n" "$dim" <<'PY'
import struct, sys
path, n, dim = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "r+b") as f:
    f.seek(0)
    f.write(struct.pack("<II", n, dim))
PY
}

# ----------------------------------------------------------------------------
# print_fbin_header file
# ----------------------------------------------------------------------------
print_fbin_header() {
    local f="$1"
    python3 - "$f" <<'PY'
import os, struct, sys
p = sys.argv[1]
sz = os.path.getsize(p)
with open(p, "rb") as f:
    n, dim = struct.unpack("<II", f.read(8))
print(f"[HDR] {p}  n={n}  dim={dim}  size={sz}")
PY
}

# ============================================================================
# SIFT-1B base (BIGANN bvecs, 132 GB)  /dev/shm/sift1b/base_1b.bin (u8bin)
# ============================================================================
download_sift1b() {
    local DEST="$SIFT_DIR/base_1b.bin"
    if [ -s "$DEST" ] && [ "$(stat -c %s "$DEST")" -ge $((100 * 1024 * 1024 * 1024)) ]; then
        log "[sift1b]  >100G"
        return
    fi
    log "[sift1b]  SIFT-1B base~132 GB..."

    local RAW="$SIFT_DIR/bigann_base.bvecs"
    if [ ! -s "$RAW" ]; then
        local URL_HF="https://huggingface.co/datasets/qbo-odp/sift-1b/resolve/main/bigann_base.bvecs"
        local URL_FTP="ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz"

        log "[sift1b]  HF mirror"
        if aria2c -x 16 -s 16 -c -d "$SIFT_DIR" -o bigann_base.bvecs "$URL_HF"; then
            log "[sift1b] HF "
        else
            log "[sift1b] HF  BIGANN FTP (gzip)"
            aria2c -x 8 -s 8 -c -d "$SIFT_DIR" -o bigann_base.bvecs.gz "$URL_FTP"
            log "[sift1b]  ..."
            gunzip "$SIFT_DIR/bigann_base.bvecs.gz"
        fi
    fi

    log "[sift1b]  bvecs  DiskANN u8bin ($DEST)"
    python3 "$(dirname "$0")/convert_bvecs_to_u8bin.py" "$RAW" "$DEST" --expected-n 1000000000

    log "[sift1b]  bvecs?  rm $RAW"
}

# ============================================================================
# DEEP base (10m/50m/100m/1b)   Yandex CDN base.1B.fbin  N
# ============================================================================
download_deep_base() {
    local SCALE="$1"                        # 10m / 50m / 100m / 1b
    local FBIN="$DEEP_DIR/base_${SCALE}.fbin"
    local DEST_U8="$DEEP_DIR/base_${SCALE}.bin"

    if [ -s "$FBIN" ] && [ -s "$DEST_U8" ]; then
        log "[deep${SCALE}] fbin+u8bin"
        return
    fi

    local n
    n="$(scale_to_n "$SCALE")"
    local header_bytes=8
    local vec_bytes=$(( DEEP_DIM * 4 ))     # fp32  4B
    local body_bytes=$(( n * vec_bytes ))
    local total_bytes=$(( header_bytes + body_bytes ))

    if [ ! -s "$FBIN" ]; then
        log "[deep${SCALE}]  DEEP base  ${n} ~$(( total_bytes / (1024*1024*1024) )) GB fp32"
        parallel_range_download "$DEEP_BASE_URL" "$FBIN" "$total_bytes"
        #  headerbase.1B.fbin  (1_000_000_000, 96) (n, 96)
        if [ "$SCALE" != "1b" ]; then
            rewrite_fbin_header "$FBIN" "$n" "$DEEP_DIM"
        fi
        print_fbin_header "$FBIN"
    else
        log "[deep${SCALE}] fbin $FBIN"
    fi

    if [ "$DEEP_MAKE_U8BIN" = "1" ] && [ ! -s "$DEST_U8" ]; then
        log "[deep${SCALE}]  fbin  u8bin  scale=[${DEEP_SCALE_MIN}, ${DEEP_SCALE_MAX}]"
        python3 "$(dirname "$0")/convert_fbin_to_u8bin.py" \
                "$FBIN" "$DEST_U8" \
                --scale-min "$DEEP_SCALE_MIN" --scale-max "$DEEP_SCALE_MAX"
    fi

    log "[deep${SCALE}] done"
}

download_deep_query() {
    local FBIN="$DEEP_DIR/query.fbin"
    local DEST_U8="$DEEP_DIR/query.bin"

    if [ ! -s "$FBIN" ]; then
        log "[deep query]  DEEP query (10k  ${DEEP_DIM})"
        aria2c -x 8 -s 8 -c -d "$DEEP_DIR" -o "query.fbin" "$DEEP_QUERY_URL"
        print_fbin_header "$FBIN"
    else
        log "[deep query] fbin "
    fi

    if [ "$DEEP_MAKE_U8BIN" = "1" ] && [ ! -s "$DEST_U8" ]; then
        log "[deep query]  fbin  u8bin"
        python3 "$(dirname "$0")/convert_fbin_to_u8bin.py" \
                "$FBIN" "$DEST_U8" \
                --scale-min "$DEEP_SCALE_MIN" --scale-max "$DEEP_SCALE_MAX"
    fi
}

# ----------------------------------------------------------------------------
# download_deep_gt 10m|100m|1b
# big-ann-benchmarks GT format:
#   [num_queries:uint32][K:uint32][ids: uint32  nqK][dists: float32  nqK]
#  groundtruth_<scale>.bin pipeline
#  preflight  gen_gt_fast.py
# ----------------------------------------------------------------------------
download_deep_gt() {
    local SCALE="$1"
    local URL=""
    case "$SCALE" in
        10m)  URL="$DEEP_GT_10M_URL" ;;
        100m) URL="$DEEP_GT_100M_URL" ;;
        1b)   URL="$DEEP_GT_1B_URL" ;;
        *)    log "[deep gt] skip unsupported scale: $SCALE"; return 0 ;;
    esac
    local DEST="$DEEP_DIR/groundtruth_deep${SCALE}.bigann.bin"
    if [ -s "$DEST" ]; then
        log "[deep gt-${SCALE}] "
        return
    fi
    log "[deep gt-${SCALE}]  $URL"
    aria2c -x 8 -s 8 -c -d "$DEEP_DIR" -o "$(basename "$DEST")" "$URL" \
        || { log "[deep gt-${SCALE}] WARN:  gen_gt_fast.py "; return 0; }
    ls -lh "$DEST"
}

# ============================================================================
# Main
# ============================================================================
ensure_tools
mkdir -p "$SIFT_DIR" "$DEEP_DIR"

case "$WHAT" in
    all)
        download_sift1b &
        PID_SIFT=$!
        download_deep_query
        download_deep_gt 10m
        download_deep_gt 100m
        download_deep_base 10m
        download_deep_base 100m
        download_deep_base 1b &
        PID_DEEP=$!
        wait "$PID_SIFT" "$PID_DEEP"
        ;;
    sift1b)   download_sift1b ;;
    deep10m)  download_deep_query; download_deep_gt 10m;  download_deep_base 10m ;;
    deep50m)  download_deep_query;                          download_deep_base 50m ;;
    deep100m) download_deep_query; download_deep_gt 100m; download_deep_base 100m ;;
    deep1b)   download_deep_query; download_deep_gt 1b;   download_deep_base 1b ;;
    *) echo "Unknown: $WHAT"; exit 1 ;;
esac

log "all done"
ls -lh "$SIFT_DIR" "$DEEP_DIR" || true
