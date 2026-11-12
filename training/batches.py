"""Owned NumPy training batches, without per-transition Python reconstruction.

Replay stores compact, validated numeric data. Its private batch factory can
copy selected rows directly into arrays. External callers use the public
constructor, which copies and validates their arrays. DQN validates again at
its boundary because a caller may edit the arrays between updates.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from action_codec import ActionCodec
from players.neural.encoder import FlatObservationEncoder


@dataclass(frozen=True, slots=True, eq=False)
class ArrayBatch:
    """One nonempty batch with canonical, contiguous, independently owned data.

    Each row represents one Transition. States use float32, rewards float64,
    actions int64, and masks/done use bool. Fields cannot be rebound normally,
    but their arrays are writable for direct PyTorch conversion without a
    read-only-buffer warning. Editing a batch never edits replay or the arrays
    supplied to its constructor. Call validate before trusting edited data;
    DQN does this automatically before targets or optimizer work.

    This class is not an immutable snapshot or a concurrent-editing interface.
    A caller must not change its arrays during an update. Batch objects compare
    by identity; compare their arrays explicitly when checking equal contents.
    """

    states: NDArray[np.float32]
    actions: NDArray[np.int64]
    rewards: NDArray[np.float64]
    next_states: NDArray[np.float32]
    done: NDArray[np.bool_]
    masks: NDArray[np.bool_]
    next_masks: NDArray[np.bool_]

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "states", "actions", "rewards", "next_states", "done", "masks", "next_masks",
    )
    _DTYPES: ClassVar[dict[str, np.dtype]] = {
        "states": np.dtype(np.float32), "actions": np.dtype(np.int64),
        "rewards": np.dtype(np.float64), "next_states": np.dtype(np.float32),
        "done": np.dtype(np.bool_), "masks": np.dtype(np.bool_),
        "next_masks": np.dtype(np.bool_),
    }

    def __post_init__(self) -> None:
        """Require canonical dtypes, then copy and validate external inputs."""
        for name, dtype in self._DTYPES.items():
            value = getattr(self, name)
            if type(value) is not np.ndarray or value.dtype != dtype:
                raise ValueError(f"array batch {name} must be a {dtype} NumPy array")
            object.__setattr__(self, name, np.array(value, copy=True, order="C"))
        self.validate()

    def __len__(self) -> int:
        """Return the number of transition rows."""
        return self.states.shape[0]

    def validate(self) -> None:
        """Check all rows with vectorized operations, without modifying them.

        Match Transition's finite values, reward range, action legality, and
        terminal-mask/reward rules. Do not impose encoder-value bounds: valid
        observations may contain negative or unusually large coin balances.
        This validation performs no replay sampling or model operations.
        """
        for name, dtype in self._DTYPES.items():
            value = getattr(self, name)
            if type(value) is not np.ndarray or value.dtype != dtype:
                raise ValueError(f"array batch {name} must be a {dtype} NumPy array")
            if not value.flags.c_contiguous or not value.flags.writeable:
                raise ValueError(f"array batch {name} must be contiguous and writable")
        if self.states.ndim != 2 or self.states.shape[1] != FlatObservationEncoder.SIZE or not len(self.states):
            raise ValueError("array batch states must have shape [batch, 332] with a positive batch size")
        count = len(self.states)
        shapes = {
            "next_states": (count, FlatObservationEncoder.SIZE),
            "actions": (count,), "rewards": (count,), "done": (count,),
            "masks": (count, ActionCodec.ACTION_COUNT),
            "next_masks": (count, ActionCodec.ACTION_COUNT),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"array batch {name} must have shape {shape}")
        if not np.isfinite(self.states).all() or not np.isfinite(self.next_states).all():
            raise ValueError("array batch states must contain finite float32 values")
        if not np.isfinite(self.rewards).all() or np.any((self.rewards < -1.5) | (self.rewards > 1.5)):
            raise ValueError("array batch rewards must be finite and between -1.5 and 1.5")
        if np.any((self.actions < 0) | (self.actions >= ActionCodec.ACTION_COUNT)):
            raise ValueError("array batch actions must be integers between 0 and 40")
        if not self.masks[np.arange(count), self.actions].all():
            raise ValueError("array batch actions must be legal under their current masks")
        if np.any(self.next_masks.any(axis=1) == self.done):
            raise ValueError("array batch next masks must match terminal flags")
        if np.any(self.rewards[~self.done] != 0):
            raise ValueError("array batch intermediate rewards must be zero")

    @classmethod
    def _from_replay(cls, **arrays: np.ndarray) -> "ArrayBatch":
        """Take ownership of canonical arrays gathered from validated replay.

        Only internal replay code uses this path. Each array must already own
        its contiguous selected data, with no views into replay or caller data.
        Avoid a second copy and duplicate content validation here; the learner
        still validates the writable batch at its public boundary.
        """
        result = object.__new__(cls)
        for name in cls.FIELD_NAMES:
            object.__setattr__(result, name, arrays[name])
        return result
