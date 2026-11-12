"""Transactionally append evaluation episodes to an existing format-2 archive.

The dataset's transition tables retain their original format. A new reward
scheme label is committed with the first explicitly timeout-aware block;
existing placement-only records and ledger fingerprints remain unchanged.
Two additive ledger tables identify each evaluation by (run_id, block_id),
where block_id normally counts completed training games. A committed block can
be recovered after a crash without evaluating or appending that block again.
Demonstration eligibility belongs to the caller, never to this storage layer.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import struct

from .dataset import (
    LEGACY_REWARD_SCHEME, Partition, _check_episode, _check_metadata, _pack_state,
    _unpack_mask, _unpack_state,
)
from .rewards import REWARD_SCHEME
from .transitions import Transition


LEDGER_VERSION = 1
ArchivedEpisodes = tuple[tuple[Partition, tuple[Transition, ...]], ...]
_COLUMNS = "state, action_id, reward, next_state, done, mask, next_mask"
_TRANSITION_SIGNATURE = (
    ("id", "INTEGER", 0, None, 1), ("state", "BLOB", 1, None, 0),
    ("action_id", "INTEGER", 1, None, 0), ("reward", "REAL", 1, None, 0),
    ("next_state", "BLOB", 1, None, 0), ("done", "INTEGER", 1, None, 0),
    ("mask", "BLOB", 1, None, 0), ("next_mask", "BLOB", 1, None, 0),
)
_LEDGER_SIGNATURES = {
    "evaluation_blocks": (
        ("run_id", "TEXT", 1, None, 1), ("block_id", "INTEGER", 1, None, 2),
        ("ledger_version", "INTEGER", 1, None, 0),
        ("model_fingerprint", "TEXT", 1, None, 0),
        ("content_fingerprint", "TEXT", 1, None, 0),
        ("metadata_json", "TEXT", 1, None, 0),
        ("episode_count", "INTEGER", 1, None, 0),
    ),
    "evaluation_episodes": (
        ("run_id", "TEXT", 1, None, 1), ("block_id", "INTEGER", 1, None, 2),
        ("episode_index", "INTEGER", 1, None, 3),
        ("partition", "TEXT", 1, None, 0),
        ("first_row_id", "INTEGER", 1, None, 0),
        ("last_row_id", "INTEGER", 1, None, 0),
        ("transition_count", "INTEGER", 1, None, 0),
    ),
}


@dataclass(frozen=True)
class ArchivedEvaluation:
    """A validated block in original evaluation-episode order.

    metadata is a fresh JSON-compatible dictionary owned by the caller.
    inserted is True only for the append that first commits this block; reads
    and identical retries return False. It does not describe replay insertion:
    checkpoint/recovery code separately controls when live buffers receive it.
    """

    run_id: str
    block_id: int
    model_fingerprint: str
    episodes: ArchivedEpisodes
    metadata: dict
    content_fingerprint: str
    inserted: bool = False


def _check_key(run_id: str, block_id: int) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    if type(block_id) is not int or not 0 <= block_id <= 2**63 - 1:
        raise ValueError("block_id must be a nonnegative SQLite integer")


def _check_model_fingerprint(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model_fingerprint must be a nonempty string")


def _metadata_json(metadata: dict) -> str:
    """Own metadata through finite JSON, preserving string keys without coercion."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON-compatible dictionary")

    def check_keys(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("metadata dictionary keys must be strings")
            for child in value.values():
                check_keys(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_keys(child)

    try:
        check_keys(metadata)
        return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("metadata must contain finite JSON-compatible values") from error


def _normalize_episodes(episodes) -> ArchivedEpisodes:
    if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)) or not episodes:
        raise ValueError("episodes must be a nonempty sequence of partition/episode pairs")
    owned = []
    for pair in episodes:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each episode must have an explicit partition and trajectory")
        partition, transitions = pair
        if partition not in ("demo", "non_demo"):
            raise ValueError("partition must be demo or non_demo")
        try:
            transitions = tuple(transitions)
        except TypeError as error:
            raise ValueError("an episode must contain Transition objects") from error
        _check_episode(transitions)
        owned.append((partition, transitions))
    return tuple(owned)


def _fingerprint(run_id, block_id, model_fingerprint, episodes, metadata_json) -> str:
    """Hash framed binary content; partition and episode order are significant."""
    digest = sha256(b"startups.evaluation_archive.v1\0")

    def add(value: bytes):
        digest.update(struct.pack("<Q", len(value)))
        digest.update(value)

    for text in (run_id, str(block_id), model_fingerprint, metadata_json):
        add(text.encode("utf-8"))
    add(struct.pack("<Q", len(episodes)))
    for partition, transitions in episodes:
        add(partition.encode("ascii"))
        add(struct.pack("<Q", len(transitions)))
        for item in transitions:
            add(_pack_state(item.state))
            # SQLite may normalize negative zero rewards to positive zero.
            reward = 0.0 if item.reward == 0 else item.reward
            add(struct.pack("<Bd?", item.action_id, reward, item.done))
            add(_pack_state(item.next_state))
            add(bytes(item.mask))
            add(bytes(item.next_mask))
    return digest.hexdigest()


def _check_table(connection, table, signature) -> None:
    kind = connection.execute("SELECT type FROM sqlite_master WHERE name=?", (table,)).fetchone()
    if kind != ("table",):
        raise ValueError(f"archive is missing the expected {table} table")
    # All identifiers come from this module's fixed schema definitions.
    actual = tuple(row[1:] for row in connection.execute(f"PRAGMA table_info({table})"))
    if actual != signature:
        raise ValueError(f"incompatible archive table schema: {table}")


def _check_archive(connection) -> bool:
    """Validate dataset/ledger schemas and metadata; return whether a ledger exists."""
    _check_table(connection, "metadata", (("key", "TEXT", 0, None, 1), ("value", "TEXT", 1, None, 0)))
    try:
        metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
        _check_metadata(metadata)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid archive metadata") from error
    for partition in ("demo", "non_demo"):
        _check_table(connection, partition, _TRANSITION_SIGNATURE)
    present = [connection.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone() is not None
               for table in _LEDGER_SIGNATURES]
    if any(present) and not all(present):
        raise ValueError("archive contains an incomplete evaluation ledger schema")
    if all(present):
        for table, signature in _LEDGER_SIGNATURES.items():
            _check_table(connection, table, signature)
    return all(present)


def _transition(row) -> Transition:
    state, action_id, reward, next_state, done, mask, next_mask = row
    if type(done) is not int or done not in (0, 1):
        raise ValueError("invalid stored done value")
    return Transition(_unpack_state(state), action_id, reward, _unpack_state(next_state),
                      bool(done), _unpack_mask(mask), _unpack_mask(next_mask))


def _check_tails(connection) -> None:
    """Check complete valid partition tails without rescanning historical games."""
    for partition in ("demo", "non_demo"):
        row = connection.execute(f"SELECT {_COLUMNS} FROM {partition} ORDER BY id DESC LIMIT 1").fetchone()
        if row is not None and not _transition(row).done:
            raise ValueError(f"{partition} ends with an incomplete episode")


def _create_ledger(connection) -> None:
    connection.execute("""CREATE TABLE evaluation_blocks (
        run_id TEXT NOT NULL, block_id INTEGER NOT NULL,
        ledger_version INTEGER NOT NULL CHECK(ledger_version = 1),
        model_fingerprint TEXT NOT NULL, content_fingerprint TEXT NOT NULL,
        metadata_json TEXT NOT NULL, episode_count INTEGER NOT NULL CHECK(episode_count > 0),
        PRIMARY KEY(run_id, block_id)
    )""")
    connection.execute("""CREATE TABLE evaluation_episodes (
        run_id TEXT NOT NULL, block_id INTEGER NOT NULL, episode_index INTEGER NOT NULL CHECK(episode_index >= 0),
        partition TEXT NOT NULL CHECK(partition IN ('demo', 'non_demo')),
        first_row_id INTEGER NOT NULL CHECK(first_row_id > 0),
        last_row_id INTEGER NOT NULL CHECK(last_row_id >= first_row_id),
        transition_count INTEGER NOT NULL CHECK(transition_count > 0),
        PRIMARY KEY(run_id, block_id, episode_index),
        FOREIGN KEY(run_id, block_id) REFERENCES evaluation_blocks(run_id, block_id)
    )""")


def _read_block(connection, run_id: str, block_id: int) -> ArchivedEvaluation | None:
    row = connection.execute("""SELECT ledger_version, model_fingerprint, content_fingerprint,
        metadata_json, episode_count FROM evaluation_blocks WHERE run_id=? AND block_id=?""", (run_id, block_id)).fetchone()
    if row is None:
        if connection.execute("SELECT 1 FROM evaluation_episodes WHERE run_id=? AND block_id=? LIMIT 1", (run_id, block_id)).fetchone():
            raise ValueError("evaluation ledger has orphaned episode records")
        return None
    version, model_fingerprint, fingerprint, metadata_json, count = row
    if type(version) is not int or version != LEDGER_VERSION or type(count) is not int or count <= 0:
        raise ValueError("incompatible or corrupt evaluation block metadata")
    _check_model_fingerprint(model_fingerprint)
    try:
        metadata_json = _metadata_json(json.loads(metadata_json))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid evaluation block JSON metadata") from error
    ranges = connection.execute("""SELECT episode_index, partition, first_row_id, last_row_id, transition_count
        FROM evaluation_episodes WHERE run_id=? AND block_id=? ORDER BY episode_index""", (run_id, block_id)).fetchall()
    if len(ranges) != count:
        raise ValueError("evaluation block episode count does not match its ledger")
    episodes = []
    previous_end = {"demo": 0, "non_demo": 0}
    for expected_index, (index, partition, first, last, length) in enumerate(ranges):
        if (type(index) is not int or index != expected_index or partition not in previous_end
                or any(type(value) is not int for value in (first, last, length))
                or first <= previous_end[partition] or last < first or length != last - first + 1):
            raise ValueError("invalid or overlapping evaluation episode row ranges")
        before = connection.execute(f"SELECT done FROM {partition} WHERE id < ? ORDER BY id DESC LIMIT 1", (first,)).fetchone()
        if before is not None and before != (1,):
            raise ValueError("evaluation episode starts inside an earlier episode")
        records = connection.execute(f"SELECT id, {_COLUMNS} FROM {partition} WHERE id BETWEEN ? AND ? ORDER BY id", (first, last)).fetchall()
        if len(records) != length or any(record[0] != first + offset for offset, record in enumerate(records)):
            raise ValueError("evaluation episode row range contains missing records")
        transitions = tuple(_transition(record[1:]) for record in records)
        _check_episode(transitions)
        episodes.append((partition, transitions))
        previous_end[partition] = last
    episodes = tuple(episodes)
    expected = _fingerprint(run_id, block_id, model_fingerprint, episodes, metadata_json)
    if fingerprint != expected:
        raise ValueError("evaluation block content fingerprint does not match stored data")
    return ArchivedEvaluation(run_id, block_id, model_fingerprint, episodes, json.loads(metadata_json), expected)


def _insert_episode(connection, partition: Partition, transitions: tuple[Transition, ...]) -> tuple[int, int]:
    """Append one already validated episode and return its inclusive row bounds."""
    first = last = None
    for item in transitions:
        cursor = connection.execute(
            f"INSERT INTO {partition} ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_pack_state(item.state), item.action_id, item.reward, _pack_state(item.next_state),
             int(item.done), bytes(item.mask), bytes(item.next_mask)),
        )
        last = cursor.lastrowid
        if first is None:
            first = last
    return first, last


def append_evaluation_block(
    path: str | Path, *, run_id: str, block_id: int, model_fingerprint: str,
    episodes: Sequence[tuple[Partition, tuple[Transition, ...]]], metadata: dict,
) -> ArchivedEvaluation:
    """Append complete episodes and their recovery ledger in one transaction.

    Open an existing archive only. Check metadata, schemas, and both partition
    tails before writing. Validate all supplied episodes, preserve their explicit
    partitions, and never modify old transitions. An explicitly timeout-aware block upgrades
    the archive's reward_scheme label in this same transaction. Old archives and
    old block retries remain readable without relabeling historical outcomes.
    BEGIN IMMEDIATE
    serializes concurrent writers, so identical retries commit no duplicate rows.

    An existing key must match the model fingerprint, episodes, and JSON metadata
    exactly; conflicting reuse raises ValueError. Any failure rolls back the
    entire block, including newly created ledger tables. This storage layer does
    not decide evaluation eligibility or update live replay/checkpoint state.
    """
    _check_key(run_id, block_id)
    _check_model_fingerprint(model_fingerprint)
    episodes = _normalize_episodes(episodes)
    metadata_json = _metadata_json(metadata)
    reward_scheme = metadata.get("reward_scheme")
    if reward_scheme not in (None, LEGACY_REWARD_SCHEME, REWARD_SCHEME):
        raise ValueError("unsupported evaluation reward scheme")
    fingerprint = _fingerprint(run_id, block_id, model_fingerprint, episodes, metadata_json)
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        has_ledger = _check_archive(connection)
        _check_tails(connection)
        if not has_ledger:
            _create_ledger(connection)
        existing = _read_block(connection, run_id, block_id)
        if existing is not None:
            if existing.content_fingerprint != fingerprint:
                raise ValueError("evaluation block key already exists with different model, episodes, or metadata")
            connection.commit()
            return existing
        if reward_scheme == REWARD_SCHEME:
            connection.execute("UPDATE metadata SET value=? WHERE key='reward_scheme'",
                               (json.dumps(REWARD_SCHEME),))
        connection.execute("INSERT INTO evaluation_blocks VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (run_id, block_id, LEDGER_VERSION, model_fingerprint, fingerprint, metadata_json, len(episodes)))
        for index, (partition, transitions) in enumerate(episodes):
            first, last = _insert_episode(connection, partition, transitions)
            connection.execute("INSERT INTO evaluation_episodes VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (run_id, block_id, index, partition, first, last, len(transitions)))
        archived = _read_block(connection, run_id, block_id)
        connection.commit()
        return replace(archived, inserted=True)
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_evaluation_block(path: str | Path, *, run_id: str, block_id: int) -> ArchivedEvaluation | None:
    """Read one committed block for crash recovery, without creating a file/table.

    Return None when this valid archive has no ledger or no matching block.
    Validate its version, JSON metadata, ordered row ranges, complete connected
    episodes, and content fingerprint. A read transaction keeps all these reads
    consistent with one database snapshot while another writer may append.
    """
    _check_key(run_id, block_id)
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
    try:
        connection.execute("BEGIN")
        if not _check_archive(connection):
            return None
        return _read_block(connection, run_id, block_id)
    finally:
        connection.close()
