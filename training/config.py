"""Validated settings for the first DQN, independent of a training framework.

This module declares the agreed experiment. It does not create a model, run an
optimizer, save a checkpoint, or resolve a device. The runner uses the same
configuration for collection, replay, evaluation, and recovery checkpoints.
"""

from dataclasses import asdict, dataclass, fields
import json
from math import isfinite
import re
from typing import ClassVar, Literal

from action_codec import ActionCodec
from players.neural.encoder import FlatObservationEncoder


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """The first experiment's settings, with configurable run and replay limits.

    The network dimensions, optimizer name and loss identify this version of the
    experiment. Capacities, batch sizes, schedules and numeric hyperparameters
    may be changed explicitly for experiments and short integration tests.
    Evaluation and checkpoint intervals count completed training games; target
    updates count optimizer updates across games. Recovery checkpoints are due
    at every checkpoint interval, regardless of whether evaluation improves.

    The opponent-pool and replay-refresh settings control the runner. Growing mode
    admits a frozen model only when its evaluation mean is strictly greater
    than opponent_admission_threshold. This is separate from promotion_threshold,
    which controls whether evaluation games enter the transition archive.
    replay_refresh_interval_games counts completed training games; None disables
    replacement of both live buffers with fresh samples from that archive.

    cpu_threads controls PyTorch's process-wide CPU intra-operation thread count.
    It does not create parallel games or change the learning schedule. The
    training entry points apply it before loading models or starting play.

    algorithm selects how nonterminal training targets are calculated: ordinary
    DQN uses the target network's legal maximum; Double DQN selects the next
    legal action with the online network and evaluates it with the target.
    This setting does not change the network architecture or inference policy.

    clip_targets optionally clamps finite TD targets to the game's possible
    return range [-1.5, 1.5], after reward and discount are applied. Raw network
    outputs and action selection are unchanged; no extra loss is added. It is
    disabled by default and when reading older configurations.

    architecture_version selects the Q-network: 1 is the original Linear/ReLU
    layout, while 2 inserts LayerNorm before each hidden ReLU. Widths, observation
    encoding and action IDs are unchanged. Changing architecture requires a new
    model, not a continuation of existing weights and optimizer state.

    q_monitor_sample_size enables optional evaluation-time Q diagnostics when
    set to a positive observation count. None disables the additional inference.
    The monitor samples replay independently of training and does not update it.
    """

    VERSION: ClassVar[int] = 6
    _VERSION_2_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "opponent_pool_mode", "opponent_admission_threshold",
        "replay_refresh_interval_games",
    })
    _VERSION_3_FIELDS: ClassVar[frozenset[str]] = frozenset({"cpu_threads"})
    _VERSION_4_FIELDS: ClassVar[frozenset[str]] = frozenset({"algorithm"})
    _VERSION_5_FIELDS: ClassVar[frozenset[str]] = frozenset({"architecture_version", "q_monitor_sample_size"})
    _VERSION_6_FIELDS: ClassVar[frozenset[str]] = frozenset({"clip_targets"})
    NETWORK_LAYERS: ClassVar[tuple[int, ...]] = (332, 256, 256, 128, 41)
    OPTIMIZER: ClassVar[str] = "AdamW"
    LOSS: ClassVar[str] = "mse"

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 1.0
    target_tau: float = 0.1
    target_update_interval: int = 1_000
    epsilon_start: float = 0.5
    epsilon_end: float = 0.01
    epsilon_decay_games: int = 10_000
    training_games: int = 10_000
    evaluation_interval_games: int = 200
    evaluation_games: int = 200
    checkpoint_interval_games: int = 200
    promotion_threshold: float = 0.45
    opponent_pool_mode: Literal["fixed", "growing"] = "fixed"
    opponent_admission_threshold: float = 0.6
    replay_refresh_interval_games: int | None = 5_000
    max_actions_per_game: int = 10_000
    demo_capacity: int = 20_000
    non_demo_capacity: int = 40_000
    demo_batch_size: int = 16
    non_demo_batch_size: int = 16
    seed: int | None = None
    device: str = "auto"
    cpu_threads: int = 1
    algorithm: Literal["dqn", "double_dqn"] = "dqn"
    architecture_version: int = 1
    q_monitor_sample_size: int | None = None
    clip_targets: bool = False

    def __post_init__(self) -> None:
        if type(self.clip_targets) is not bool:
            raise ValueError("clip_targets must be a boolean")
        if type(self.architecture_version) is not int or self.architecture_version not in (1, 2):
            raise ValueError("architecture_version must be 1 or 2")
        if self.q_monitor_sample_size is not None:
            _positive_integer(self.q_monitor_sample_size, "q_monitor_sample_size")
        if type(self.algorithm) is not str or self.algorithm not in ("dqn", "double_dqn"):
            raise ValueError("algorithm must be dqn or double_dqn")
        for name in (
            "target_update_interval", "epsilon_decay_games", "training_games",
            "evaluation_interval_games", "evaluation_games", "checkpoint_interval_games",
            "max_actions_per_game", "demo_capacity", "non_demo_capacity",
            "demo_batch_size", "non_demo_batch_size", "cpu_threads",
        ):
            _positive_integer(getattr(self, name), name)
        if self.replay_refresh_interval_games is not None:
            _positive_integer(self.replay_refresh_interval_games, "replay_refresh_interval_games")
        if type(self.opponent_pool_mode) is not str or self.opponent_pool_mode not in ("fixed", "growing"):
            raise ValueError("opponent_pool_mode must be fixed or growing")
        for name in (
            "learning_rate", "weight_decay", "gamma", "target_tau",
            "epsilon_start", "epsilon_end", "promotion_threshold",
            "opponent_admission_threshold",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise ValueError(f"{name} must be a finite number")
            try:
                value = float(value)
            except OverflowError as error:
                raise ValueError(f"{name} must be a finite number") from error
            if not isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, value)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie between zero and one")
        if not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must lie above zero and at most one")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not -1.5 <= self.promotion_threshold <= 1.5:
            raise ValueError("promotion_threshold must lie between -1.5 and 1.5")
        if not -1.5 <= self.opponent_admission_threshold <= 1.5:
            raise ValueError("opponent_admission_threshold must lie between -1.5 and 1.5")
        if self.demo_batch_size > self.demo_capacity or self.non_demo_batch_size > self.non_demo_capacity:
            raise ValueError("each replay capacity must fit its requested batch size")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None")
        if not isinstance(self.device, str) or re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", self.device) is None:
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")

    @property
    def batch_size(self) -> int:
        """Return the combined update batch size; defaults to 16 + 16 = 32."""
        return self.demo_batch_size + self.non_demo_batch_size

    def epsilon_for_game(self, completed_training_games: int) -> float:
        """Return exploration at the next game's start, clamped after decay ends.

        Evaluation games do not contribute to this counter. The initial game
        sees epsilon_start; epsilon_end is reached after epsilon_decay_games
        completed training games and retained thereafter.
        """
        if type(completed_training_games) is not int or completed_training_games < 0:
            raise ValueError("completed_training_games must be a nonnegative integer")
        if completed_training_games >= self.epsilon_decay_games:
            return self.epsilon_end
        fraction = completed_training_games / self.epsilon_decay_games
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def to_dict(self) -> dict[str, object]:
        """Return fresh JSON-compatible settings plus encoder/model format metadata."""
        return {
            "config_version": self.VERSION,
            "network_layers": list(self.NETWORK_LAYERS),
            "optimizer": self.OPTIMIZER,
            "loss": self.LOSS,
            "observation_encoding": FlatObservationEncoder.metadata(),
            "action_codec_version": ActionCodec.VERSION,
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "TrainingConfig":
        """Restore version 6 or migrate a complete version-1 through version-5 record.

        Version 1 predates growing pools and periodic replay replacement. It
        loads with fixed opponents, admission threshold 0.6 and refresh disabled
        to preserve its behavior. Versions 1 and 2 use cpu_threads=1, the new
        default for this small network. Versions 1 through 3 use ordinary DQN,
        preserving their original target calculation. Versions 1 through 4 use
        architecture version 1, preserving their original parameter layout, and
        leave optional Q monitoring disabled.
        Versions 1 through 5 leave target clipping disabled. Saving any restored
        record writes version 6. Missing, mixed-version and unknown fields are rejected;
        the input is never edited. Parsing does not itself change PyTorch's
        runtime settings.
        """
        if not isinstance(values, dict):
            raise ValueError("configuration must be a dictionary")
        version = values.get("config_version")
        if type(version) is not int or version not in (1, 2, 3, 4, 5, cls.VERSION):
            raise ValueError("incompatible configuration metadata: config_version")
        names = {item.name for item in fields(cls)}
        expected = cls().to_dict()
        if version < 6:
            for name in cls._VERSION_6_FIELDS:
                del expected[name]
        if version < 5:
            for name in cls._VERSION_5_FIELDS:
                del expected[name]
        if version < 4:
            for name in cls._VERSION_4_FIELDS:
                del expected[name]
        if version < 3:
            for name in cls._VERSION_3_FIELDS:
                del expected[name]
        if version == 1:
            for name in cls._VERSION_2_FIELDS:
                del expected[name]
        expected["config_version"] = version
        if set(values) != set(expected):
            raise ValueError("configuration fields do not match this format")
        for name in set(expected) - names:
            try:
                actual_metadata = json.dumps(values[name], sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid configuration metadata: {name}") from error
            # JSON comparison distinguishes True from version 1, including
            # nested encoder metadata, unlike ordinary Python equality.
            if actual_metadata != json.dumps(expected[name], sort_keys=True):
                raise ValueError(f"incompatible configuration metadata: {name}")
        settings = {name: values[name] for name in names if name in values}
        if version < 6:
            settings["clip_targets"] = False
        if version < 5:
            settings["architecture_version"] = 1
            settings["q_monitor_sample_size"] = None
        if version < 4:
            settings["algorithm"] = "dqn"
        if version < 3:
            settings["cpu_threads"] = 1
        if version == 1:
            settings.update(
                opponent_pool_mode="fixed",
                opponent_admission_threshold=0.6,
                replay_refresh_interval_games=None,
            )
        return cls(**settings)
