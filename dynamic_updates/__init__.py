"""Snapshot-delta update prototype for InSituANN.

This package is intentionally isolated from the production CUDA runners.  It
models the update semantics we want for online inserts/deletes:

* immutable main IVF snapshot
* mutable delta overlay
* tombstone deletion
* private rebuild followed by atomic state swap
"""

from .snapshot_delta_index import (
    ActiveState,
    DeltaIndex,
    MainSnapshot,
    ResidualPQEncoder,
    RoutingStageOutput,
    SearchResult,
    SnapshotDeltaIndex,
    TombstoneBitmap,
    build_snapshot,
)

__all__ = [
    "ActiveState",
    "DeltaIndex",
    "MainSnapshot",
    "ResidualPQEncoder",
    "RoutingStageOutput",
    "SearchResult",
    "SnapshotDeltaIndex",
    "TombstoneBitmap",
    "build_snapshot",
]
