"""Compact FIFO replay buffers and read-only initialization from a dataset.

The two pools are disjoint by source partition. Sampling never consumes a
transition or infers its partition from reward. The archive stores completed
episodes; replay samples individual transitions for the first, one-step DQN.
The existing Transition API uses only the standard library. The array-sampling
path imports NumPy lazily; neither path owns an optimizer or writes to disk.
"""

from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from random import Random
import sys
from typing import TYPE_CHECKING

from action_codec import ActionCodec
from players.neural.encoder import FlatObservationEncoder

from .config import TrainingConfig, _positive_integer
from .dataset import Partition
from .transitions import Transition

if TYPE_CHECKING:
    from .batches import ArrayBatch


def _check_seed(seed: int | None) -> None:
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer or None")


class ReplayBuffer:
    """A fixed-capacity FIFO ring with a private, reproducible sampling RNG.

    Own compact CPU storage: float32 states, float64 rewards (preserving the
    Transition reward exactly), and byte-valued action IDs, done flags and masks.
    Appending copies data and evicts the oldest transition once full. Reading
    or sampling returns fresh immutable Transition objects, never storage views.
    No full observations, hidden game state, or policy objects are retained.
    """

    _STATE_SIZE = FlatObservationEncoder.SIZE
    _MASK_SIZE = ActionCodec.ACTION_COUNT
    STATE_VERSION = 1

    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        _positive_integer(capacity, "capacity")
        _check_seed(seed)
        if array("f").itemsize != 4 or array("d").itemsize != 8:
            raise RuntimeError("replay requires four-byte floats and eight-byte doubles")
        self._capacity = capacity
        self._rng = Random(seed)
        self._size = 0
        self._head = 0
        # Indirection permits reservoir initialization to restore FIFO order
        # without copying a second capacity-sized collection of state vectors.
        self._order = list(range(capacity))
        self._states = array("f", [0.0]) * (capacity * self._STATE_SIZE)
        self._next_states = array("f", [0.0]) * (capacity * self._STATE_SIZE)
        self._rewards = array("d", [0.0]) * capacity
        self._actions = bytearray(capacity)
        self._done = bytearray(capacity)
        self._masks = bytearray(capacity * self._MASK_SIZE)
        self._next_masks = bytearray(capacity * self._MASK_SIZE)

    @property
    def capacity(self) -> int:
        """Return the maximum number of retained transitions."""
        return self._capacity

    @property
    def storage_nbytes(self) -> int:
        """Return allocated numeric payload bytes, excluding small Python bookkeeping."""
        return (
            (len(self._states) + len(self._next_states)) * 4
            + len(self._rewards) * 8 + len(self._actions) + len(self._done)
            + len(self._masks) + len(self._next_masks)
        )

    def __len__(self) -> int:
        return self._size

    def reseed(self, seed: int | None = None) -> None:
        """Reseed sampling when supplied an integer; None preserves the RNG stream.

        Do not remove transitions or change FIFO order. Seed affects sampling,
        never the order in which appended transitions are stored or evicted.
        """
        _check_seed(seed)
        if seed is not None:
            self._rng.seed(seed)

    def _slot(self, index: int) -> int:
        return self._order[(self._head + index) % self._capacity]

    def _write(self, slot: int, transition: Transition) -> None:
        state_start = slot * self._STATE_SIZE
        mask_start = slot * self._MASK_SIZE
        self._states[state_start:state_start + self._STATE_SIZE] = array("f", transition.state)
        self._next_states[state_start:state_start + self._STATE_SIZE] = array("f", transition.next_state)
        self._rewards[slot] = transition.reward
        self._actions[slot] = transition.action_id
        self._done[slot] = transition.done
        self._masks[mask_start:mask_start + self._MASK_SIZE] = bytes(transition.mask)
        self._next_masks[mask_start:mask_start + self._MASK_SIZE] = bytes(transition.next_mask)

    def append(self, transition: Transition) -> None:
        """Copy one validated Transition, evicting the oldest entry if full."""
        if not isinstance(transition, Transition):
            raise ValueError("replay entries must be Transition objects")
        if self._size < self._capacity:
            self._write(self._slot(self._size), transition)
            self._size += 1
        else:
            self._write(self._slot(0), transition)
            self._head = (self._head + 1) % self._capacity

    def extend(self, transitions: Iterable[Transition]) -> None:
        """Validate the input collection first, then append it in supplied order.

        A bad entry or an iterator error leaves this buffer unchanged. Whole
        game/partition eligibility remains the caller's responsibility. FIFO
        limits count individual transitions, so eviction may split an episode.
        """
        transitions = tuple(transitions)
        if any(not isinstance(item, Transition) for item in transitions):
            raise ValueError("replay entries must be Transition objects")
        for transition in transitions:
            self.append(transition)

    def __getitem__(self, index: int) -> Transition:
        """Read a chronological entry (zero is oldest), including negative indices."""
        if type(index) is not int:
            raise TypeError("replay indices must be integers")
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError("replay index out of range")
        slot = self._slot(index)
        state_start = slot * self._STATE_SIZE
        mask_start = slot * self._MASK_SIZE
        return Transition(
            tuple(self._states[state_start:state_start + self._STATE_SIZE]),
            self._actions[slot], self._rewards[slot],
            tuple(self._next_states[state_start:state_start + self._STATE_SIZE]),
            bool(self._done[slot]),
            tuple(bool(value) for value in self._masks[mask_start:mask_start + self._MASK_SIZE]),
            tuple(bool(value) for value in self._next_masks[mask_start:mask_start + self._MASK_SIZE]),
        )

    def __iter__(self) -> Iterator[Transition]:
        """Yield oldest to newest; callers must not mutate the buffer mid-iteration."""
        for index in range(self._size):
            yield self[index]

    def snapshot(self) -> tuple[Transition, ...]:
        """Materialize an immutable chronological copy, independent of later eviction."""
        return tuple(self)

    def sample(self, batch_size: int) -> tuple[Transition, ...]:
        """Uniformly sample entries without replacement within this batch.

        The same entry may appear in subsequent batches. Equal-valued records
        at different positions remain separate experiences. Sampling neither
        consumes data nor changes FIFO order. Reject unavailable sizes before
        advancing the private RNG.
        """
        _positive_integer(batch_size, "batch_size")
        if batch_size > self._size:
            raise ValueError("batch_size exceeds the number of available transitions")
        return tuple(self[index] for index in self._rng.sample(range(self._size), batch_size))

    def sample_arrays(self, batch_size: int) -> "ArrayBatch":
        """Sample the same rows as sample(), directly into owned NumPy arrays.

        Draw the same logical indices with the same private RNG progression,
        then gather their physical ring slots. Source storage was validated on
        insertion or checkpoint loading, so sampling need not rebuild and
        revalidate Transition objects. The batch owns its selected data: later
        FIFO writes, resizes, or batch edits cannot affect the other side.
        No checkpoint format or public Transition-returning method changes.
        """
        _positive_integer(batch_size, "batch_size")
        if batch_size > self._size:
            raise ValueError("batch_size exceeds the number of available transitions")
        return self._gather_arrays(self._rng.sample(range(self._size), batch_size))

    def arrays_at(self, indices) -> "ArrayBatch":
        """Copy specified chronological rows without consuming the replay RNG.

        This supports diagnostic sampling independently of training. Zero means
        the oldest occupied row, even after FIFO wraparound. Require a nonempty
        sequence of in-range integer indices; repeated indices are allowed.
        Returned arrays own their storage, just like sample_arrays().
        """
        indices = tuple(indices)
        if not indices or any(type(index) is not int or not 0 <= index < self._size for index in indices):
            raise ValueError("indices must be nonempty in-range chronological integers")
        return self._gather_arrays(indices)

    def _gather_arrays(self, indices) -> "ArrayBatch":
        """Gather already validated logical indices into owned arrays."""
        import numpy as np
        from .batches import ArrayBatch

        slots = np.fromiter(
            (self._slot(index) for index in indices), dtype=np.intp, count=len(indices),
        )
        # Advanced indexing copies each selection instead of returning a view
        # into the mutable compact ring. Only action IDs need a dtype change.
        return ArrayBatch._from_replay(
            states=np.frombuffer(self._states, dtype=np.float32).reshape(self._capacity, self._STATE_SIZE)[slots],
            actions=np.frombuffer(self._actions, dtype=np.uint8)[slots].astype(np.int64),
            rewards=np.frombuffer(self._rewards, dtype=np.float64)[slots],
            next_states=np.frombuffer(self._next_states, dtype=np.float32).reshape(self._capacity, self._STATE_SIZE)[slots],
            done=np.frombuffer(self._done, dtype=np.bool_)[slots],
            masks=np.frombuffer(self._masks, dtype=np.bool_).reshape(self._capacity, self._MASK_SIZE)[slots],
            next_masks=np.frombuffer(self._next_masks, dtype=np.bool_).reshape(self._capacity, self._MASK_SIZE)[slots],
        )

    def state_dict(self) -> dict[str, object]:
        """Snapshot compact chronological bytes and the next sampling RNG state.

        Serialize only occupied slots, not unused capacity. Canonical little-
        endian float32 states and float64 rewards are portable across machines.
        Physical ring positions are deliberately omitted: chronological order
        plus RNG state preserves both future FIFO eviction and sampled entries.
        No tensors, pickle-only objects or expanded transition tuples are used.
        """
        def numeric(storage: array, width: int) -> bytes:
            packed = array(storage.typecode)
            for index in range(self._size):
                start = self._slot(index) * width
                packed.extend(storage[start:start + width])
            if sys.byteorder != "little":
                packed.byteswap()
            return packed.tobytes()

        def byte_values(storage: bytearray, width: int) -> bytes:
            packed = bytearray()
            for index in range(self._size):
                start = self._slot(index) * width
                packed.extend(storage[start:start + width])
            return bytes(packed)

        return {
            "format_version": self.STATE_VERSION, "byte_order": "little",
            "observation_encoding": FlatObservationEncoder.metadata(),
            "action_codec_version": ActionCodec.VERSION,
            "capacity": self._capacity, "size": self._size,
            "states": numeric(self._states, self._STATE_SIZE),
            "next_states": numeric(self._next_states, self._STATE_SIZE),
            "rewards": numeric(self._rewards, 1),
            "actions": byte_values(self._actions, 1),
            "done": byte_values(self._done, 1),
            "masks": byte_values(self._masks, self._MASK_SIZE),
            "next_masks": byte_values(self._next_masks, self._MASK_SIZE),
            "rng_state": self._rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ReplayBuffer":
        """Restore a versioned compact snapshot after validating every record.

        Check byte lengths, finite numbers, action legality, boolean masks,
        terminal masks/rewards, schema metadata and the sampling RNG before
        allocating capacity. Returned storage is independent of the snapshot.
        """
        expected = {"format_version", "byte_order", "observation_encoding", "action_codec_version",
                    "capacity", "size", "states", "next_states", "rewards", "actions",
                    "done", "masks", "next_masks", "rng_state"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("replay snapshot fields do not match this format")
        if type(state["format_version"]) is not int or state["format_version"] != cls.STATE_VERSION:
            raise ValueError("unsupported replay snapshot version")
        if state["byte_order"] != "little":
            raise ValueError("replay snapshots must use little-endian encoding")
        # Reuse configuration's strict JSON metadata comparison, which also
        # distinguishes booleans from integer format/version identifiers.
        metadata = TrainingConfig().to_dict()
        metadata["observation_encoding"] = state["observation_encoding"]
        metadata["action_codec_version"] = state["action_codec_version"]
        TrainingConfig.from_dict(metadata)
        _positive_integer(state["capacity"], "capacity")
        size = state["size"]
        if type(size) is not int or not 0 <= size <= state["capacity"]:
            raise ValueError("replay snapshot size must fit its capacity")
        widths = {"states": cls._STATE_SIZE * 4, "next_states": cls._STATE_SIZE * 4,
                  "rewards": 8, "actions": 1, "done": 1,
                  "masks": cls._MASK_SIZE, "next_masks": cls._MASK_SIZE}
        for name, width in widths.items():
            if type(state[name]) is not bytes or len(state[name]) != size * width:
                raise ValueError(f"invalid replay byte payload: {name}")
        rng = Random()
        try:
            rng.setstate(state["rng_state"])
        except (TypeError, ValueError, OverflowError, IndexError) as error:
            raise ValueError("invalid replay sampling RNG state") from error
        numeric = {}
        for name, code in (("states", "f"), ("next_states", "f"), ("rewards", "d")):
            values = array(code)
            values.frombytes(state[name])
            if sys.byteorder != "little":
                values.byteswap()
            if any(not isfinite(value) for value in values):
                raise ValueError(f"replay {name} must contain finite numbers")
            numeric[name] = values
        if any(value not in (0, 1) for name in ("done", "masks", "next_masks") for value in state[name]):
            raise ValueError("replay flags and masks must contain only zero or one")
        for index, action in enumerate(state["actions"]):
            start = index * cls._MASK_SIZE
            mask = state["masks"][start:start + cls._MASK_SIZE]
            next_mask = state["next_masks"][start:start + cls._MASK_SIZE]
            done = state["done"][index]
            reward = numeric["rewards"][index]
            if action >= cls._MASK_SIZE or not mask[action]:
                raise ValueError("replay action must be legal under its current mask")
            if bool(any(next_mask)) == bool(done):
                raise ValueError("replay next mask does not match its terminal flag")
            if not -1.5 <= reward <= 1.5 or (not done and reward != 0):
                raise ValueError("invalid replay transition reward")
        result = cls(state["capacity"])
        result._size = size
        result._rng = rng
        for name in ("states", "next_states", "rewards"):
            getattr(result, f"_{name}")[:len(numeric[name])] = numeric[name]
        for name in ("actions", "done", "masks", "next_masks"):
            getattr(result, f"_{name}")[:len(state[name])] = state[name]
        return result


@dataclass(frozen=True)
class ReplayBuffers:
    """Independent demo and non_demo pools with their configured batch composition.

    archive_counts records rows seen during initialization or refresh, before
    capacity sampling. It remains a load summary as new transitions arrive.
    Empty archive_counts means these buffers were constructed directly.
    """

    demo: ReplayBuffer
    non_demo: ReplayBuffer
    demo_batch_size: int = 16
    non_demo_batch_size: int = 16
    archive_counts: tuple[tuple[Partition, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.demo, ReplayBuffer) or not isinstance(self.non_demo, ReplayBuffer):
            raise ValueError("demo and non_demo must be ReplayBuffer objects")
        if self.demo is self.non_demo:
            raise ValueError("demo and non_demo must be distinct replay buffers")
        for name in ("demo", "non_demo"):
            count = getattr(self, f"{name}_batch_size")
            _positive_integer(count, f"{name}_batch_size")
            if count > getattr(self, name).capacity:
                raise ValueError("a pool's batch size cannot exceed its capacity")

    def sample_batch(self) -> tuple[Transition, ...]:
        """Return demo samples followed by non_demo samples, 16 + 16 by default.

        Validate both pools before either RNG advances. Source identity follows
        the pool, not reward. In particular, non_demo may contain outright wins.
        Tensor conversion and loss calculation belong to the DQN update engine.
        """
        if len(self.demo) < self.demo_batch_size or len(self.non_demo) < self.non_demo_batch_size:
            raise ValueError("both replay pools must contain their requested batch sizes")
        return self.demo.sample(self.demo_batch_size) + self.non_demo.sample(self.non_demo_batch_size)

    def sample_array_batch(self) -> "ArrayBatch":
        """Return demo then non_demo rows as one contiguous, owned array batch.

        This is the training fast path; sample_batch retains the original tuple
        interface for callers inspecting Transition objects. Both paths consume
        exactly the same RNG draws and preserve the configured partition mix.
        Check both occupancies before either private sampling RNG advances.
        """
        if len(self.demo) < self.demo_batch_size or len(self.non_demo) < self.non_demo_batch_size:
            raise ValueError("both replay pools must contain their requested batch sizes")
        import numpy as np
        from .batches import ArrayBatch

        demo = self.demo.sample_arrays(self.demo_batch_size)
        non_demo = self.non_demo.sample_arrays(self.non_demo_batch_size)
        return ArrayBatch._from_replay(**{
            name: np.concatenate((getattr(demo, name), getattr(non_demo, name)), axis=0)
            for name in ArrayBatch.FIELD_NAMES
        })

    def state_dict(self) -> dict[str, object]:
        """Snapshot both disjoint pools, their batch sizes and archive load counts."""
        return {"format_version": 1, "demo": self.demo.state_dict(),
                "non_demo": self.non_demo.state_dict(), "demo_batch_size": self.demo_batch_size,
                "non_demo_batch_size": self.non_demo_batch_size, "archive_counts": self.archive_counts}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "ReplayBuffers":
        """Restore independently owned pools with their exact future sample streams."""
        expected = {"format_version", "demo", "non_demo", "demo_batch_size",
                    "non_demo_batch_size", "archive_counts"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("replay-pair snapshot fields do not match this format")
        if type(state["format_version"]) is not int or state["format_version"] != 1:
            raise ValueError("unsupported replay-pair snapshot version")
        counts = state["archive_counts"]
        if not isinstance(counts, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 or item[0] not in ("demo", "non_demo")
            or type(item[1]) is not int or item[1] < 0 for item in counts
        ) or len({item[0] for item in counts}) != len(counts):
            raise ValueError("invalid replay archive counts")
        return cls(ReplayBuffer.from_state_dict(state["demo"]),
                   ReplayBuffer.from_state_dict(state["non_demo"]),
                   state["demo_batch_size"], state["non_demo_batch_size"], counts)


def reconfigure_buffers(buffers: ReplayBuffers, config: TrainingConfig) -> ReplayBuffers:
    """Apply replay capacities and batch sizes without discarding experience.

    Validate both pools before allocating or changing anything. Resized pools
    own independent compact storage and preserve chronological order, exact
    payload values and their next sampling RNG state. An unchanged capacity
    reuses the original pool, including its RNG, to avoid copying large replay
    arrays; subsequent operations on that pool are therefore shared between
    the old and new wrappers. This call itself never mutates either source
    pool or consumes randomness. config.seed does not reseed recovered replay.
    Archive counts remain the most recent archive-load summary.
    """
    if not isinstance(buffers, ReplayBuffers):
        raise ValueError("buffers must be ReplayBuffers")
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    for partition in ("demo", "non_demo"):
        size = len(getattr(buffers, partition))
        if getattr(config, f"{partition}_capacity") < size:
            raise ValueError(f"{partition}_capacity cannot be below current occupancy ({size})")
        if getattr(config, f"{partition}_batch_size") > size:
            raise ValueError(f"{partition}_batch_size exceeds current occupancy ({size})")

    pools = {}
    for partition in ("demo", "non_demo"):
        pool = getattr(buffers, partition)
        capacity = getattr(config, f"{partition}_capacity")
        if capacity != pool.capacity:
            state = pool.state_dict()
            state["capacity"] = capacity
            pool = ReplayBuffer.from_state_dict(state)
        pools[partition] = pool
    return ReplayBuffers(
        pools["demo"], pools["non_demo"], config.demo_batch_size,
        config.non_demo_batch_size, buffers.archive_counts,
    )


def _sample_transitions(
    transitions: Iterable[Transition], capacity: int,
    *, selection_rng: Random, sampling_seed: int,
) -> tuple[ReplayBuffer, int]:
    """Reservoir-sample a validated chronological stream in bounded storage."""
    buffer = ReplayBuffer(capacity, seed=sampling_seed)
    positions: list[int] = []
    total = 0
    for index, transition in enumerate(transitions):
        total += 1
        if index < capacity:
            buffer.append(transition)
            positions.append(index)
        else:
            candidate = selection_rng.randrange(index + 1)
            if candidate < capacity:
                buffer._write(buffer._slot(candidate), transition)
                positions[candidate] = index
    # Reservoir slots are unordered. Sort slot references, not the vectors, to
    # preserve the archive's age order before FIFO insertion/eviction resumes.
    buffer._order[:len(buffer)] = sorted(range(len(buffer)), key=positions.__getitem__)
    return buffer, total


def load_initial_buffers(
    path: str | Path, config: TrainingConfig | None = None, *, seed: int | None = None,
) -> ReplayBuffers:
    """Load each SQLite partition independently, uniformly up to its capacity.

    Select row IDs before reading transition values, giving every row equal
    inclusion probability without loading the entire archive into Python
    objects. Check only selected transitions and skip invalid ones without
    drawing replacements. Retain valid selections in chronological FIFO order.
    Both partitions are read from one transaction, with archive metadata and
    schema checked first. This is not a full-archive or episode-continuity audit.

    The explicit seed overrides config.seed; if neither is specified, use local
    system-seeded randomness. Selection and each pool's subsequent batch sampling
    use separate RNG streams, with no changes to Python's global random state.
    Return total archive row counts as ReplayBuffers.archive_counts, including
    invalid rows. A small archive or a sample containing invalid records can
    load successfully even when it has insufficient entries for a batch;
    sample_batch then reports that condition without changing either RNG.
    This opens the archive read-only and never appends, edits, or deduplicates it.
    """
    config = TrainingConfig() if config is None else config
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    _check_seed(seed)
    # Lazy imports keep module initialization independent of the sampler,
    # which constructs ReplayBuffer objects, and of refresh orchestration.
    from .archive_sampling import sample_archive_partition
    from .replay_refresh import _bounds, _open_snapshot

    rng = Random(config.seed if seed is None else seed)
    pools = {}
    counts = []
    connection = _open_snapshot(Path(path).resolve())
    try:
        for partition, capacity in (("demo", config.demo_capacity), ("non_demo", config.non_demo_capacity)):
            selection_rng = Random(rng.getrandbits(64))
            pool, total = sample_archive_partition(
                connection, partition, bounds=_bounds(connection, partition), capacity=capacity,
                selection_rng=selection_rng, sampling_seed=rng.getrandbits(64),
            )
            pools[partition] = pool
            counts.append((partition, total))
    finally:
        connection.close()
    return ReplayBuffers(
        pools["demo"], pools["non_demo"], config.demo_batch_size, config.non_demo_batch_size, tuple(counts),
    )
