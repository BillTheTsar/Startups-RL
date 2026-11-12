"""Replace both replay buffers from a reproducible read of the shared archive.

The runner records archive bounds before a refresh, then calls refresh_buffers.
Both tables are sampled from one read transaction. A retry with the saved bounds
and RNG state selects the same rows even if another run has appended examples.
This relies on the archive's append-only contract: saved rows must not be edited
or replaced. Nothing here schedules refreshes or changes existing RAM buffers.
"""

from pathlib import Path
from random import Random
import sqlite3

from .archive import _COLUMNS, _check_archive, _transition
from .archive_sampling import sample_archive_partition
from .config import TrainingConfig
from .dataset import _whole_episodes
from .replay import ReplayBuffers, _sample_transitions


_PARTITIONS = ("demo", "non_demo")


def _open_snapshot(path: Path) -> sqlite3.Connection:
    """Start a read-only transaction shared by every query in this operation."""
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None)
    try:
        connection.execute("BEGIN")
        _check_archive(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def _bounds(connection, partition: str, limit: int | None = None) -> dict[str, int]:
    """Count a whole table or its recorded prefix in the current transaction."""
    suffix, parameters = ("", ()) if limit is None else (" WHERE id <= ?", (limit,))
    count, first, last = connection.execute(
        f"SELECT COUNT(*), MIN(id), MAX(id) FROM {partition}{suffix}", parameters,
    ).fetchone()
    if count and (first <= 0 or count > last):
        raise ValueError("archive row IDs must be positive")
    return {"row_count": count, "max_row_id": 0 if last is None else last}


def prepare_archive_refresh(path: str | Path) -> dict:
    """Return JSON-ready bounds for both partitions from one database snapshot.

    Persist this record before replacing RAM buffers. The maximum row IDs keep
    later appends out of a retry; row counts detect missing records. Empty
    partitions can be described here, but cannot satisfy a training batch.
    Validate archive schemas and row IDs without reading transition payloads.
    Version 2 records use selected-only validation when their refresh runs.
    """
    path = Path(path).resolve()
    connection = _open_snapshot(path)
    try:
        partitions = {name: _bounds(connection, name) for name in _PARTITIONS}
        return {"version": 2, "archive_path": str(path), "partitions": partitions}
    finally:
        connection.close()


def validate_refresh_snapshot(snapshot: dict, path: str | Path) -> None:
    """Check the saved bounds' shape, types and archive path without disk access."""
    if type(snapshot) is not dict or set(snapshot) != {"version", "archive_path", "partitions"}:
        raise ValueError("invalid archive refresh snapshot fields")
    if type(snapshot["version"]) is not int or snapshot["version"] not in (1, 2):
        raise ValueError("unsupported archive refresh snapshot version")
    if type(snapshot["archive_path"]) is not str or snapshot["archive_path"] != str(Path(path).resolve()):
        raise ValueError("refresh snapshot belongs to a different archive path")
    partitions = snapshot["partitions"]
    if type(partitions) is not dict or set(partitions) != set(_PARTITIONS):
        raise ValueError("refresh snapshot must contain both archive partitions")
    for values in partitions.values():
        if type(values) is not dict or set(values) != {"row_count", "max_row_id"} or any(
            type(value) is not int or not 0 <= value <= 2**63 - 1 for value in values.values()
        ):
            raise ValueError("invalid refresh row bounds")
        count, last = values["row_count"], values["max_row_id"]
        if count > last or (count == 0) != (last == 0):
            raise ValueError("refresh row count and bound disagree")


def refresh_buffers(
    path: str | Path, config: TrainingConfig, *, rng: Random, snapshot: dict,
) -> ReplayBuffers:
    """Build independent replacement buffers, advancing rng only after success.

    Select rows uniformly without replacement within each partition, up to its
    configured capacity. Keep selected rows in archive order for later FIFO
    eviction. For current snapshots, read and validate only selected transition
    payloads. Skip invalid selected rows without replacements; require enough
    valid rows for each configured batch. Whole-episode chains are not checked
    because a sample need not contain adjacent transitions. Legacy version-1
    pending refreshes retain their original full-scan validation and sampling.

    The supplied RNG is the runner's separate, saved refresh stream. Selection
    and each new buffer's batch RNG use seeds drawn from that stream, rather
    than restarting config.seed at every refresh. A failed read or validation
    leaves it unchanged. The caller replaces its live pair only after return;
    ordinary RAM-only experience is deliberately absent from the result.
    """
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    if not isinstance(rng, Random):
        raise ValueError("rng must be a Random instance")
    path = Path(path).resolve()
    validate_refresh_snapshot(snapshot, path)
    for name in _PARTITIONS:
        if snapshot["partitions"][name]["row_count"] < getattr(config, f"{name}_batch_size"):
            raise ValueError(f"archive refresh cannot supply one {name} batch")
    working_rng = Random()
    working_rng.setstate(rng.getstate())
    connection = _open_snapshot(path)
    try:
        # Check both bounds before allocating replacement buffers. All later
        # reads remain in this same transaction even if another writer appends.
        for name in _PARTITIONS:
            expected = snapshot["partitions"][name]
            if _bounds(connection, name, expected["max_row_id"]) != expected:
                raise ValueError(f"archive {name} prefix differs from the refresh snapshot")
        pools, counts = {}, []
        for name in _PARTITIONS:
            selection_rng = Random(working_rng.getrandbits(64))
            sampling_seed = working_rng.getrandbits(64)
            if snapshot["version"] == 1:
                # Preserve an already-journaled refresh's old behavior exactly,
                # including its physical replay slots and future batch samples.
                cursor = connection.execute(
                    f"SELECT {_COLUMNS} FROM {name} WHERE id <= ? ORDER BY id",
                    (snapshot["partitions"][name]["max_row_id"],),
                )
                transitions = _whole_episodes(_transition(row) for row in cursor)
                pool, total = _sample_transitions(
                    transitions, getattr(config, f"{name}_capacity"),
                    selection_rng=selection_rng, sampling_seed=sampling_seed,
                )
            else:
                pool, total = sample_archive_partition(
                    connection, name, bounds=snapshot["partitions"][name],
                    capacity=getattr(config, f"{name}_capacity"),
                    selection_rng=selection_rng, sampling_seed=sampling_seed,
                )
            if len(pool) < getattr(config, f"{name}_batch_size"):
                raise ValueError(f"archive refresh cannot supply one {name} batch after validation")
            pools[name] = pool
            counts.append((name, total))
        result = ReplayBuffers(pools["demo"], pools["non_demo"], config.demo_batch_size,
                               config.non_demo_batch_size, tuple(counts))
    finally:
        connection.close()
    rng.setstate(working_rng.getstate())
    return result
