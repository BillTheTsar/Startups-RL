"""Small, model-independent SQLite datasets of completed game transitions.

The demo table supplies demonstration experience; non_demo supplies the separate
ordinary replay pool. Partition selection belongs to the collector, not the
reward: a winning model episode may still be non_demo. States are little-endian
float32 blobs, masks are 41 boolean bytes,
and rewards are SQLite REAL values. No torch, NumPy, pickle, replay sampling,
or training-loop implementation is needed to create/read this file.
"""

from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import sys
from typing import Literal

from action_codec import ActionCodec
from players.neural.encoder import FlatObservationEncoder

from .rewards import REWARD_SCHEME
from .transitions import Transition


Partition = Literal["demo", "non_demo"]
FORMAT_VERSION = 2
_METADATA = {
    "format_version": FORMAT_VERSION,
    "observation_encoding": FlatObservationEncoder.metadata(),
    "action_codec_version": ActionCodec.VERSION,
    "action_count": ActionCodec.ACTION_COUNT,
    "state_byte_order": "little",
    "reward_scheme": REWARD_SCHEME,
}
LEGACY_REWARD_SCHEME = "placement_points_v1"


def _check_metadata(metadata: dict) -> None:
    """Accept the exact current or legacy contract without rewriting the file.

    Normal placement rewards have the same meaning in both schemes. The newer
    label additionally permits a -1 terminal timeout outcome; stored historical
    transitions retain their original rewards. All encoder, codec, schema and
    numeric metadata must still match exactly, including integer versus boolean.
    """
    if not isinstance(metadata, dict) or metadata.get("reward_scheme") not in (
        LEGACY_REWARD_SCHEME, REWARD_SCHEME,
    ):
        raise ValueError("dataset metadata has an unsupported reward scheme")
    expected = dict(_METADATA, reward_scheme=metadata["reward_scheme"])
    try:
        actual_json = json.dumps(metadata, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid dataset metadata") from error
    if actual_json != json.dumps(expected, sort_keys=True):
        raise ValueError("dataset metadata is incompatible with this encoder/codec/format")


def _check_episode(transitions: tuple[Transition, ...]) -> None:
    """Require a whole connected episode without inferring demonstration quality."""
    if not transitions or any(not isinstance(item, Transition) for item in transitions):
        raise ValueError("an episode must contain Transition objects")
    if not transitions[-1].done or any(item.done for item in transitions[:-1]):
        raise ValueError("an episode must end with exactly one terminal transition")
    for before, after in zip(transitions, transitions[1:]):
        if before.next_state != after.state or before.next_mask != after.mask:
            raise ValueError("consecutive transitions must share a decision state and mask")


@dataclass
class TransitionDataset:
    """Demonstration and other experience, populated with complete games.

    add_episode appends the entire focal trajectory to the caller's partition.
    Demonstration eligibility is a collection policy, independent of reward;
    non_demo accepts winning, tied, and losing model episodes alike.
    The file loader iterates records instead of materializing this container,
    so future replay buffers can be populated without loading the file twice.
    """

    demo: list[Transition] = field(default_factory=list)
    non_demo: list[Transition] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.demo) + len(self.non_demo)

    def add_episode(self, transitions: Iterable[Transition], *, partition: Partition) -> None:
        """Validate a full trajectory, then append to explicit demo or non_demo.

        The caller decides whether this experience qualifies as a demonstration.
        Never move an episode between partitions based on its terminal reward.
        """
        if partition not in ("demo", "non_demo"):
            raise ValueError("partition must be demo or non_demo")
        transitions = tuple(transitions)
        _check_episode(transitions)
        getattr(self, partition).extend(transitions)


def _pack_state(state) -> bytes:
    values = array("f", state)
    if values.itemsize != 4:
        raise ValueError("float32 storage requires four-byte C floats")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _unpack_state(blob: bytes) -> tuple[float, ...]:
    if type(blob) is not bytes or len(blob) != FlatObservationEncoder.SIZE * 4:
        raise ValueError("invalid state blob; expected 332 float32 entries")
    values = array("f")
    if values.itemsize != 4:
        raise ValueError("float32 storage requires four-byte C floats")
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(values)


def _unpack_mask(blob: bytes) -> tuple[bool, ...]:
    if type(blob) is not bytes or len(blob) != ActionCodec.ACTION_COUNT or any(value not in (0, 1) for value in blob):
        raise ValueError("invalid mask blob; expected 41 bytes containing 0 or 1")
    return tuple(bool(value) for value in blob)


def _whole_episodes(items: Iterable[Transition]) -> Iterator[Transition]:
    """Validate episode boundaries when saving or streaming an entire partition."""
    pending = []
    for item in items:
        if not isinstance(item, Transition):
            raise ValueError("dataset entries must be Transition objects")
        pending.append(item)
        if item.done:
            _check_episode(tuple(pending))
            yield from pending
            pending.clear()
    if pending:
        raise ValueError("partition ends with an incomplete episode")


def save_dataset(path: str | Path, dataset: TransitionDataset) -> None:
    """Write both partitions and format metadata to a new SQLite file.

    Refuse to overwrite an existing file. All rows belong to a single database
    transaction; remove this newly created file on failure. Minimal schema
    metadata identifies encoder/codec compatibility, not a run audit system.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb"):
        pass
    connection = None
    try:
        connection = sqlite3.connect(path)
        with connection:
            connection.execute("BEGIN")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", ((key, json.dumps(value)) for key, value in _METADATA.items()))
            for partition in ("demo", "non_demo"):
                # Table identifiers are fixed here, never supplied by a caller.
                connection.execute(f"""CREATE TABLE {partition} (
                    id INTEGER PRIMARY KEY,
                    state BLOB NOT NULL CHECK(length(state) = 1328),
                    action_id INTEGER NOT NULL CHECK(action_id BETWEEN 0 AND 40),
                    reward REAL NOT NULL CHECK(reward BETWEEN -1.5 AND 1.5),
                    next_state BLOB NOT NULL CHECK(length(next_state) = 1328),
                    done INTEGER NOT NULL CHECK(done IN (0, 1)),
                    mask BLOB NOT NULL CHECK(length(mask) = 41),
                    next_mask BLOB NOT NULL CHECK(length(next_mask) = 41)
                )""")
                records = _whole_episodes(getattr(dataset, partition))
                connection.executemany(
                    f"INSERT INTO {partition} (state, action_id, reward, next_state, done, mask, next_mask) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ((_pack_state(item.state), item.action_id, item.reward,
                      _pack_state(item.next_state), int(item.done), bytes(item.mask), bytes(item.next_mask))
                     for item in records),
                )
    except BaseException:
        if connection is not None:
            connection.close()
            connection = None
        path.unlink(missing_ok=True)
        raise
    finally:
        if connection is not None:
            connection.close()


def iter_transitions(path: str | Path, partition: Partition | None = None) -> Iterator[Transition]:
    """Read and validate transitions, retaining their order within each partition.

    None yields demo followed by non_demo for whole-dataset inspection. Replay
    loading uses a separate selected-row reader. Labels reflect collection
    policy; rewards do not determine or restrict a partition. Episodes end
    with done=True. This is storage order, not randomized replay sampling or
    chronological ordering across the two partitions. Open the file read-only
    and reject incompatible metadata, corrupt masks/vectors, or partial games.
    Both original placement-only archives and timeout-aware archives are read
    without changing their metadata or historical records.
    """
    if partition not in (None, "demo", "non_demo"):
        raise ValueError("partition must be demo, non_demo, or None")
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
        _check_metadata(metadata)
        for table in ("demo", "non_demo") if partition is None else (partition,):
            cursor = connection.execute(f"SELECT state, action_id, reward, next_state, done, mask, next_mask FROM {table} ORDER BY id")

            def records() -> Iterator[Transition]:
                for state, action_id, reward, next_state, done, mask, next_mask in cursor:
                    if type(done) is not int or done not in (0, 1):
                        raise ValueError("invalid stored done value")
                    yield Transition(_unpack_state(state), action_id, reward, _unpack_state(next_state),
                                     bool(done), _unpack_mask(mask), _unpack_mask(next_mask))

            yield from _whole_episodes(records())
    finally:
        connection.close()
