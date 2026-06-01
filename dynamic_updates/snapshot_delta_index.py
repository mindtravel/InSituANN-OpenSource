"""Isolated snapshot-delta update layer prototype.

This file implements the exact semantics discussed for online updates without
touching the existing CUDA search code:

* Main IVF data is immutable after publication.
* Inserts go to a host-authoritative delta store plus an optional PQ-code view.
* Deletes are represented by tombstones and skipped before candidate selection.
* Rebuild/compaction constructs a private snapshot and publishes it atomically.

The implementation uses NumPy and small synthetic indexes so that behavior can
be checked locally.  It is not intended to replace the production kernels; it
is a reference model for integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SearchResult:
    ids: Array
    distances: Array


@dataclass(frozen=True)
class RoutingStageOutput:
    """Compact metadata returned by the simulated GPU stage.

    ``main_probes`` corresponds to the top-nprobe IVF centroids for the main
    immutable snapshot.  ``delta_positions`` / ``delta_ids`` are the bounded
    top candidates from the GPU-resident delta index.  The host stage consumes
    both outputs: it scans only the main IVF lists and exact-reranks those
    candidates together with the returned delta candidates.
    """

    main_probes: Array
    delta_positions: Tuple[Array, ...]
    delta_ids: Tuple[Array, ...]
    delta_scores: Tuple[Array, ...]


def _as_float32(vectors: Array) -> Array:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {arr.shape}")
    return arr


def _as_int64(ids: Array) -> Array:
    arr = np.asarray(ids, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"ids must be 1-D, got shape {arr.shape}")
    return arr


def _l2_squared(a: Array, b: Array) -> Array:
    diff = a[:, None, :] - b[None, :, :]
    return np.einsum("nkd,nkd->nk", diff, diff, optimize=True)


def _topk_from_scores(ids: Array, scores: Array, k: int) -> SearchResult:
    if ids.size == 0:
        return SearchResult(
            ids=np.empty(0, dtype=np.int64),
            distances=np.empty(0, dtype=np.float32),
        )
    take = min(k, ids.size)
    order = np.argpartition(scores, take - 1)[:take]
    order = order[np.argsort(scores[order], kind="stable")]
    return SearchResult(ids=ids[order].astype(np.int64), distances=scores[order].astype(np.float32))


def _assign_nearest(data: Array, centroids: Array, batch_size: int = 2048) -> Array:
    data = _as_float32(data)
    centroids = _as_float32(centroids)
    assignments = np.empty(data.shape[0], dtype=np.int32)
    for start in range(0, data.shape[0], batch_size):
        end = min(start + batch_size, data.shape[0])
        dist = _l2_squared(data[start:end], centroids)
        assignments[start:end] = np.argmin(dist, axis=1).astype(np.int32)
    return assignments


def _partition_dim(dim: int, m: int) -> List[slice]:
    if m <= 0 or m > dim:
        raise ValueError(f"invalid PQ M={m} for dim={dim}")
    base = dim // m
    rem = dim % m
    parts: List[slice] = []
    start = 0
    for i in range(m):
        width = base + (1 if i < rem else 0)
        parts.append(slice(start, start + width))
        start += width
    return parts


def _kmeans(data: Array, k: int, iters: int, seed: int) -> Array:
    """Small deterministic k-means used by synthetic benchmarks."""
    data = _as_float32(data)
    n = data.shape[0]
    if n == 0:
        raise ValueError("cannot run kmeans on empty data")
    k = min(k, n)
    rng = np.random.default_rng(seed)
    if n >= k:
        init_idx = rng.choice(n, size=k, replace=False)
    else:
        init_idx = rng.choice(n, size=k, replace=True)
    centroids = data[init_idx].copy()
    for _ in range(max(1, iters)):
        dist = _l2_squared(data, centroids)
        assign = np.argmin(dist, axis=1)
        for cid in range(k):
            mask = assign == cid
            if np.any(mask):
                centroids[cid] = data[mask].mean(axis=0)
    return centroids.astype(np.float32)


@dataclass(frozen=True)
class TombstoneBitmap:
    """Immutable tombstone set published inside an ActiveState."""

    deleted: frozenset[int] = field(default_factory=frozenset)

    def contains(self, ids: Array) -> Array:
        ids = _as_int64(ids)
        if not self.deleted:
            return np.zeros(ids.shape, dtype=bool)
        return np.fromiter((int(x) in self.deleted for x in ids), dtype=bool, count=ids.size)

    def add(self, ids: Iterable[int]) -> "TombstoneBitmap":
        merged = set(self.deleted)
        merged.update(int(x) for x in ids)
        return TombstoneBitmap(frozenset(merged))

    def without_missing(self, live_ids: Iterable[int]) -> "TombstoneBitmap":
        live = set(int(x) for x in live_ids)
        return TombstoneBitmap(frozenset(x for x in self.deleted if x in live))

    @property
    def count(self) -> int:
        return len(self.deleted)


@dataclass(frozen=True)
class MainSnapshot:
    ids: Array
    vectors: Array
    centroids: Array
    assignments: Array
    lists: Tuple[Array, ...]
    pq_codes: Optional[Array] = None

    def __post_init__(self) -> None:
        if self.ids.shape[0] != self.vectors.shape[0]:
            raise ValueError("ids and vectors must have the same length")
        if self.assignments.shape[0] != self.vectors.shape[0]:
            raise ValueError("assignments and vectors must have the same length")

    @property
    def nlist(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def route(self, queries: Array, nprobe: int) -> Array:
        queries = _as_float32(queries)
        dist = _l2_squared(queries, self.centroids)
        p = min(max(1, int(nprobe)), self.nlist)
        top = np.argpartition(dist, p - 1, axis=1)[:, :p]
        row = np.arange(queries.shape[0])[:, None]
        return top[row, np.argsort(dist[row, top], axis=1)]

    def candidate_positions(self, probes: Sequence[int]) -> Array:
        chunks = [self.lists[int(cid)] for cid in probes if 0 <= int(cid) < self.nlist]
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks).astype(np.int64, copy=False)


@dataclass(frozen=True)
class DeltaIndex:
    ids: Array
    vectors: Array
    mode: str = "flat"
    centroids: Optional[Array] = None
    assignments: Optional[Array] = None
    lists: Optional[Tuple[Array, ...]] = None
    pq_codes: Optional[Array] = None

    @staticmethod
    def empty(dim: int, pq_m: Optional[int] = None) -> "DeltaIndex":
        pq_codes = None
        if pq_m is not None:
            pq_codes = np.empty((0, pq_m), dtype=np.uint16)
        return DeltaIndex(
            ids=np.empty(0, dtype=np.int64),
            vectors=np.empty((0, dim), dtype=np.float32),
            pq_codes=pq_codes,
        )

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def nlist(self) -> int:
        if self.mode != "ivf" or self.centroids is None:
            return 0
        return int(self.centroids.shape[0])

    def append(self, ids: Array, vectors: Array, pq_codes: Optional[Array]) -> "DeltaIndex":
        ids = _as_int64(ids)
        vectors = _as_float32(vectors)
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("delta ids and vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError("delta vector dimensionality mismatch")
        merged_codes = self.pq_codes
        if pq_codes is not None:
            pq_codes = np.asarray(pq_codes, dtype=np.uint16)
            merged_codes = pq_codes if merged_codes is None else np.vstack([merged_codes, pq_codes])
        elif merged_codes is not None:
            pad = np.zeros((ids.shape[0], merged_codes.shape[1]), dtype=np.uint16)
            merged_codes = np.vstack([merged_codes, pad])
        return DeltaIndex(
            ids=np.concatenate([self.ids, ids]),
            vectors=np.vstack([self.vectors, vectors]),
            mode="flat",
            pq_codes=merged_codes,
        )

    def build_ivf(self, nlist_delta: int, iters: int = 5, seed: int = 0) -> "DeltaIndex":
        if self.size == 0:
            return self
        nlist = min(max(1, int(nlist_delta)), self.size)
        centroids = _kmeans(self.vectors, nlist, iters=iters, seed=seed)
        dist = _l2_squared(self.vectors, centroids)
        assignments = np.argmin(dist, axis=1).astype(np.int32)
        lists = tuple(np.flatnonzero(assignments == cid).astype(np.int64) for cid in range(nlist))
        return DeltaIndex(
            ids=self.ids.copy(),
            vectors=self.vectors.copy(),
            mode="ivf",
            centroids=centroids,
            assignments=assignments,
            lists=lists,
            pq_codes=None if self.pq_codes is None else self.pq_codes.copy(),
        )

    def candidate_positions(self, query: Array, nprobe: int) -> Array:
        if self.size == 0:
            return np.empty(0, dtype=np.int64)
        if self.mode != "ivf" or self.centroids is None or self.lists is None:
            return np.arange(self.size, dtype=np.int64)
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        dist = _l2_squared(q, self.centroids)[0]
        p = min(max(1, int(nprobe)), self.nlist)
        probes = np.argpartition(dist, p - 1)[:p]
        probes = probes[np.argsort(dist[probes], kind="stable")]
        chunks = [self.lists[int(cid)] for cid in probes]
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks).astype(np.int64, copy=False)


@dataclass(frozen=True)
class ResidualPQEncoder:
    codebooks: Tuple[Array, ...]
    parts: Tuple[slice, ...]
    version: int = 0

    @staticmethod
    def train(
        vectors: Array,
        centroids: Array,
        assignments: Array,
        m: int,
        ksub: int = 16,
        iters: int = 8,
        seed: int = 0,
    ) -> "ResidualPQEncoder":
        vectors = _as_float32(vectors)
        centroids = _as_float32(centroids)
        assignments = np.asarray(assignments, dtype=np.int64)
        residuals = vectors - centroids[assignments]
        parts = tuple(_partition_dim(vectors.shape[1], m))
        codebooks = []
        for i, part in enumerate(parts):
            cb = _kmeans(residuals[:, part], k=min(ksub, residuals.shape[0]), iters=iters, seed=seed + i)
            codebooks.append(cb)
        return ResidualPQEncoder(tuple(codebooks), parts, version=1)

    @property
    def m(self) -> int:
        return len(self.parts)

    def encode(self, vectors: Array, centroid_vectors: Array) -> Array:
        vectors = _as_float32(vectors)
        centroid_vectors = _as_float32(centroid_vectors)
        if vectors.shape != centroid_vectors.shape:
            raise ValueError("vectors and centroids must have the same shape for residual encoding")
        residuals = vectors - centroid_vectors
        codes = np.empty((vectors.shape[0], self.m), dtype=np.uint16)
        for i, part in enumerate(self.parts):
            cb = self.codebooks[i]
            dist = _l2_squared(residuals[:, part], cb)
            codes[:, i] = np.argmin(dist, axis=1).astype(np.uint16)
        return codes

    def adc_scores(self, query: Array, centroid: Array, codes: Array) -> Array:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        centroid = np.asarray(centroid, dtype=np.float32).reshape(-1)
        codes = np.asarray(codes, dtype=np.uint16)
        target_residual = query - centroid
        scores = np.zeros(codes.shape[0], dtype=np.float32)
        for i, part in enumerate(self.parts):
            cb = self.codebooks[i]
            lut = np.sum((cb - target_residual[part]) ** 2, axis=1)
            code = np.minimum(codes[:, i], cb.shape[0] - 1)
            scores += lut[code]
        return scores


@dataclass
class ActiveState:
    main_snapshot: MainSnapshot
    delta_index: DeltaIndex
    tombstones: TombstoneBitmap
    version: int = 0
    refcount: int = 0
    ref_lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def pin(self) -> "PinnedState":
        with self.ref_lock:
            self.refcount += 1
        return PinnedState(self)


class PinnedState:
    def __init__(self, state: ActiveState) -> None:
        self.state = state

    def __enter__(self) -> ActiveState:
        return self.state

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        with self.state.ref_lock:
            self.state.refcount -= 1


@dataclass(frozen=True)
class RebuildJob:
    base_state: ActiveState
    new_snapshot: MainSnapshot
    pq_encoder: Optional[ResidualPQEncoder]


def build_snapshot(
    ids: Array,
    vectors: Array,
    centroids: Array,
    pq_encoder: Optional[ResidualPQEncoder] = None,
) -> MainSnapshot:
    ids = _as_int64(ids)
    vectors = _as_float32(vectors)
    centroids = _as_float32(centroids)
    if ids.shape[0] != vectors.shape[0]:
        raise ValueError("ids and vectors length mismatch")
    assignments = _assign_nearest(vectors, centroids)
    lists = tuple(np.flatnonzero(assignments == cid).astype(np.int64) for cid in range(centroids.shape[0]))
    pq_codes = None
    if pq_encoder is not None:
        pq_codes = pq_encoder.encode(vectors, centroids[assignments])
    return MainSnapshot(
        ids=ids.copy(),
        vectors=vectors.copy(),
        centroids=centroids.copy(),
        assignments=assignments,
        lists=lists,
        pq_codes=pq_codes,
    )


class SnapshotDeltaIndex:
    """Reference implementation of the proposed online update semantics."""

    def __init__(
        self,
        main_snapshot: MainSnapshot,
        pq_encoder: Optional[ResidualPQEncoder] = None,
        delta_ratio_threshold: float = 0.05,
        delete_ratio_threshold: float = 0.10,
        nlist_growth_threshold: float = 1.25,
    ) -> None:
        self.pq_encoder = pq_encoder
        self.delta_ratio_threshold = float(delta_ratio_threshold)
        self.delete_ratio_threshold = float(delete_ratio_threshold)
        self.nlist_growth_threshold = float(nlist_growth_threshold)
        self._lock = Lock()
        self._active_state = ActiveState(
            main_snapshot=main_snapshot,
            delta_index=DeltaIndex.empty(main_snapshot.dim, pq_encoder.m if pq_encoder else None),
            tombstones=TombstoneBitmap(),
            version=0,
        )
        self._pending_rebuild: Optional[RebuildJob] = None

    def pin_state(self) -> PinnedState:
        return self._active_state.pin()

    @property
    def active_state(self) -> ActiveState:
        return self._active_state

    def insert(self, ids: Array, vectors: Array, publish: bool = True) -> DeltaIndex:
        """Append inserts to staging delta; publish atomically by default."""
        ids = _as_int64(ids)
        vectors = _as_float32(vectors)
        state = self._active_state
        if np.intersect1d(ids, state.main_snapshot.ids).size:
            raise ValueError("insert ids already exist in main snapshot")
        if np.intersect1d(ids, state.delta_index.ids).size:
            raise ValueError("insert ids already exist in delta")

        pq_codes = None
        if self.pq_encoder is not None:
            assign = np.argmin(_l2_squared(vectors, state.main_snapshot.centroids), axis=1)
            pq_codes = self.pq_encoder.encode(vectors, state.main_snapshot.centroids[assign])
        staging = state.delta_index.append(ids, vectors, pq_codes)
        if publish:
            self.publish_delta(staging)
        return staging

    @staticmethod
    def _merge_delta_indexes(current: DeltaIndex, staging: DeltaIndex) -> DeltaIndex:
        if current.size == 0:
            return staging
        if staging.size == 0:
            return current
        seen = set(int(x) for x in current.ids)
        append_mask = np.fromiter((int(x) not in seen for x in staging.ids), dtype=bool, count=staging.size)
        if not np.any(append_mask):
            return current

        pq_codes: Optional[Array] = None
        if current.pq_codes is not None or staging.pq_codes is not None:
            if current.pq_codes is None or staging.pq_codes is None:
                raise ValueError("cannot merge PQ and non-PQ delta indexes")
            pq_codes = np.vstack([current.pq_codes, staging.pq_codes[append_mask]])

        return DeltaIndex(
            ids=np.concatenate([current.ids, staging.ids[append_mask]]),
            vectors=np.vstack([current.vectors, staging.vectors[append_mask]]),
            mode="flat",
            pq_codes=pq_codes,
        )

    def publish_delta(self, staging_delta: DeltaIndex) -> None:
        with self._lock:
            old = self._active_state
            merged_delta = self._merge_delta_indexes(old.delta_index, staging_delta)
            self._active_state = ActiveState(
                main_snapshot=old.main_snapshot,
                delta_index=merged_delta,
                tombstones=old.tombstones,
                version=old.version + 1,
            )

    def delete(self, ids: Iterable[int]) -> None:
        with self._lock:
            old = self._active_state
            self._active_state = ActiveState(
                main_snapshot=old.main_snapshot,
                delta_index=old.delta_index,
                tombstones=old.tombstones.add(ids),
                version=old.version + 1,
            )

    def build_delta_ivf(self, nlist_delta: int, iters: int = 5, seed: int = 0) -> None:
        with self._lock:
            old = self._active_state
            self._active_state = ActiveState(
                main_snapshot=old.main_snapshot,
                delta_index=old.delta_index.build_ivf(nlist_delta, iters=iters, seed=seed),
                tombstones=old.tombstones,
                version=old.version + 1,
            )

    def needs_compaction(self) -> bool:
        state = self._active_state
        main_n = max(1, state.main_snapshot.size)
        delta_ratio = state.delta_index.size / main_n
        delete_ratio = state.tombstones.count / main_n
        total_nlist = state.main_snapshot.nlist + state.delta_index.nlist
        return (
            delta_ratio >= self.delta_ratio_threshold
            or delete_ratio >= self.delete_ratio_threshold
            or total_nlist > self.nlist_growth_threshold * state.main_snapshot.nlist
        )

    def route_and_delta_stage(
        self,
        queries: Array,
        nprobe: int = 8,
        delta_topk: int = 10,
        delta_score_mode: str = "exact",
        state: Optional[ActiveState] = None,
    ) -> RoutingStageOutput:
        """Simulate the fused GPU stage for routing plus delta retrieval.

        The production design should execute these together on the GPU and
        return compact metadata to the host:

        * top-nprobe main IVF centroid ids for each query
        * top-delta_topk candidates from the GPU-resident delta index

        Deleted delta ids are filtered before entering the delta top-k.
        """
        state = state or self._active_state
        queries = _as_float32(queries)
        probes = state.main_snapshot.route(queries, nprobe)
        top_positions: List[Array] = []
        top_ids: List[Array] = []
        top_scores: List[Array] = []

        for q in queries:
            delta_pos = state.delta_index.candidate_positions(q, nprobe)
            delta_ids = state.delta_index.ids[delta_pos]
            live = ~state.tombstones.contains(delta_ids)
            delta_pos = delta_pos[live]
            delta_ids = delta_ids[live]
            if delta_pos.size == 0:
                top_positions.append(np.empty(0, dtype=np.int64))
                top_ids.append(np.empty(0, dtype=np.int64))
                top_scores.append(np.empty(0, dtype=np.float32))
                continue

            if delta_score_mode == "pq":
                if self.pq_encoder is None or state.delta_index.pq_codes is None:
                    raise ValueError("delta_score_mode='pq' requires delta PQ codes")
                assign = np.argmin(_l2_squared(state.delta_index.vectors[delta_pos], state.main_snapshot.centroids), axis=1)
                scores = np.empty(delta_pos.shape[0], dtype=np.float32)
                for cid in np.unique(assign):
                    mask = assign == cid
                    scores[mask] = self.pq_encoder.adc_scores(
                        q,
                        state.main_snapshot.centroids[int(cid)],
                        state.delta_index.pq_codes[delta_pos[mask]],
                    )
            elif delta_score_mode == "exact":
                vecs = state.delta_index.vectors[delta_pos]
                scores = np.sum((vecs - q[None, :]) ** 2, axis=1).astype(np.float32)
            else:
                raise ValueError(f"unknown delta_score_mode={delta_score_mode!r}")

            local = _topk_from_scores(delta_pos, scores, max(1, int(delta_topk)))
            top_positions.append(local.ids)
            top_ids.append(state.delta_index.ids[local.ids])
            top_scores.append(local.distances)

        return RoutingStageOutput(
            main_probes=probes,
            delta_positions=tuple(top_positions),
            delta_ids=tuple(top_ids),
            delta_scores=tuple(top_scores),
        )

    def _build_compacted_snapshot(
        self,
        old: ActiveState,
        refresh_pq: bool,
    ) -> Tuple[MainSnapshot, Optional[ResidualPQEncoder]]:
        live_main_mask = ~old.tombstones.contains(old.main_snapshot.ids)
        live_delta_mask = ~old.tombstones.contains(old.delta_index.ids)

        new_ids = np.concatenate([old.main_snapshot.ids[live_main_mask], old.delta_index.ids[live_delta_mask]])
        new_vectors = np.vstack(
            [old.main_snapshot.vectors[live_main_mask], old.delta_index.vectors[live_delta_mask]]
        )

        encoder = self.pq_encoder
        if refresh_pq and encoder is not None:
            tmp_snapshot = build_snapshot(new_ids, new_vectors, old.main_snapshot.centroids)
            refreshed = ResidualPQEncoder.train(
                tmp_snapshot.vectors,
                tmp_snapshot.centroids,
                tmp_snapshot.assignments,
                m=encoder.m,
                ksub=max(cb.shape[0] for cb in encoder.codebooks),
                iters=6,
                seed=17,
            )
            encoder = ResidualPQEncoder(refreshed.codebooks, refreshed.parts, version=encoder.version + 1)

        new_snapshot = build_snapshot(new_ids, new_vectors, old.main_snapshot.centroids, encoder)
        return new_snapshot, encoder

    def compact_begin(self, refresh_pq: bool = False) -> RebuildJob:
        """Build a private compaction snapshot without publishing it.

        Queries and updates may continue against the current active state while
        this private snapshot is being built.  ``compact_finish`` merges in any
        tail updates that arrived after this cutoff.
        """
        old = self._active_state
        new_snapshot, encoder = self._build_compacted_snapshot(old, refresh_pq)
        job = RebuildJob(base_state=old, new_snapshot=new_snapshot, pq_encoder=encoder)
        self._pending_rebuild = job
        return job

    @staticmethod
    def _tail_delta(current: DeltaIndex, base: DeltaIndex) -> DeltaIndex:
        if current.size == 0:
            return DeltaIndex.empty(current.dim, current.pq_codes.shape[1] if current.pq_codes is not None else None)
        base_ids = set(int(x) for x in base.ids)
        mask = np.fromiter((int(x) not in base_ids for x in current.ids), dtype=bool, count=current.size)
        pq_codes = None if current.pq_codes is None else current.pq_codes[mask].copy()
        return DeltaIndex(
            ids=current.ids[mask].copy(),
            vectors=current.vectors[mask].copy(),
            mode="flat",
            pq_codes=pq_codes,
        )

    @staticmethod
    def _reencode_delta_with_encoder(
        delta: DeltaIndex,
        snapshot: MainSnapshot,
        encoder: Optional[ResidualPQEncoder],
    ) -> DeltaIndex:
        if encoder is None or delta.size == 0:
            return delta
        assign = np.argmin(_l2_squared(delta.vectors, snapshot.centroids), axis=1)
        pq_codes = encoder.encode(delta.vectors, snapshot.centroids[assign])
        return DeltaIndex(
            ids=delta.ids.copy(),
            vectors=delta.vectors.copy(),
            mode=delta.mode,
            centroids=None if delta.centroids is None else delta.centroids.copy(),
            assignments=None if delta.assignments is None else delta.assignments.copy(),
            lists=delta.lists,
            pq_codes=pq_codes,
        )

    def compact_finish(self, job: Optional[RebuildJob] = None, refresh_pq: Optional[bool] = None) -> None:
        """Publish a completed private compaction snapshot.

        Tail inserts/deletes that arrived after ``compact_begin`` are preserved:
        old delta vectors folded into the new snapshot are removed from the
        active delta, while later delta vectors remain in the overlay.
        """
        if refresh_pq is not None and job is None and self._pending_rebuild is None:
            job = self.compact_begin(refresh_pq=refresh_pq)
        job = job or self._pending_rebuild
        if job is None:
            raise ValueError("no pending compaction job")

        with self._lock:
            current = self._active_state
            tail_delta = self._tail_delta(current.delta_index, job.base_state.delta_index)
            if job.pq_encoder is not self.pq_encoder:
                self.pq_encoder = job.pq_encoder
                tail_delta = self._reencode_delta_with_encoder(tail_delta, job.new_snapshot, self.pq_encoder)

            if tail_delta.size:
                live_ids_iter: Iterator[int] = iter(np.concatenate([job.new_snapshot.ids, tail_delta.ids]).tolist())
            else:
                live_ids_iter = iter(job.new_snapshot.ids.tolist())
            tombstones = current.tombstones.without_missing(live_ids_iter)

            self._active_state = ActiveState(
                main_snapshot=job.new_snapshot,
                delta_index=tail_delta,
                tombstones=tombstones,
                version=current.version + 1,
            )
            if self._pending_rebuild is job:
                self._pending_rebuild = None

    def compact(self, refresh_pq: bool = False) -> None:
        """Build and publish a private snapshot, preserving tail updates."""
        job = self.compact_begin(refresh_pq=refresh_pq)
        self.compact_finish(job)

    def search_exact(
        self,
        queries: Array,
        k: int = 10,
        nprobe: int = 8,
        delta_topk: Optional[int] = None,
        state: Optional[ActiveState] = None,
    ) -> List[SearchResult]:
        state = state or self._active_state
        queries = _as_float32(queries)
        stage = self.route_and_delta_stage(
            queries,
            nprobe=nprobe,
            delta_topk=delta_topk or k,
            delta_score_mode="exact",
            state=state,
        )
        out: List[SearchResult] = []
        for qi, q in enumerate(queries):
            main_pos = state.main_snapshot.candidate_positions(stage.main_probes[qi])
            main_ids = state.main_snapshot.ids[main_pos]
            main_live = ~state.tombstones.contains(main_ids)
            main_ids = main_ids[main_live]
            main_vecs = state.main_snapshot.vectors[main_pos][main_live]

            delta_pos = stage.delta_positions[qi]
            delta_ids = stage.delta_ids[qi]
            delta_vecs = state.delta_index.vectors[delta_pos] if delta_pos.size else np.empty((0, state.main_snapshot.dim), dtype=np.float32)

            ids = np.concatenate([main_ids, delta_ids])
            if ids.size == 0:
                out.append(_topk_from_scores(ids, np.empty(0, dtype=np.float32), k))
                continue
            vecs = np.vstack([main_vecs, delta_vecs])
            scores = np.sum((vecs - q[None, :]) ** 2, axis=1)
            out.append(_topk_from_scores(ids, scores.astype(np.float32), k))
        return out

    def search_pq(
        self,
        queries: Array,
        k: int = 10,
        nprobe: int = 8,
        rerank_budget: int = 512,
        delta_topk: Optional[int] = None,
        state: Optional[ActiveState] = None,
    ) -> List[SearchResult]:
        if self.pq_encoder is None:
            raise ValueError("PQ search requires a ResidualPQEncoder")
        state = state or self._active_state
        if state.main_snapshot.pq_codes is None:
            raise ValueError("main snapshot does not have PQ codes")
        queries = _as_float32(queries)
        stage = self.route_and_delta_stage(
            queries,
            nprobe=nprobe,
            delta_topk=delta_topk or k,
            delta_score_mode="pq" if state.delta_index.pq_codes is not None else "exact",
            state=state,
        )
        out: List[SearchResult] = []
        for qi, q in enumerate(queries):
            cand_ids: List[Array] = []
            cand_vecs: List[Array] = []

            for cid in stage.main_probes[qi]:
                pos = state.main_snapshot.lists[int(cid)]
                if pos.size == 0:
                    continue
                ids = state.main_snapshot.ids[pos]
                live = ~state.tombstones.contains(ids)
                if not np.any(live):
                    continue
                live_pos = pos[live]
                scores = self.pq_encoder.adc_scores(
                    q,
                    state.main_snapshot.centroids[int(cid)],
                    state.main_snapshot.pq_codes[live_pos],
                )
                local = _topk_from_scores(live_pos, scores, rerank_budget)
                cand_ids.append(state.main_snapshot.ids[local.ids])
                cand_vecs.append(state.main_snapshot.vectors[local.ids])

            delta_pos = stage.delta_positions[qi]
            if delta_pos.size:
                cand_ids.append(stage.delta_ids[qi])
                cand_vecs.append(state.delta_index.vectors[delta_pos])

            if not cand_ids:
                out.append(SearchResult(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
                continue
            ids = np.concatenate(cand_ids)
            vecs = np.vstack(cand_vecs)
            final_live = ~state.tombstones.contains(ids)
            ids = ids[final_live]
            vecs = vecs[final_live]
            scores = np.sum((vecs - q[None, :]) ** 2, axis=1)
            out.append(_topk_from_scores(ids, scores.astype(np.float32), k))
        return out
