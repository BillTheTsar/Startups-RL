"""Explicit opponent admission, separate from scheduling and training recovery.

This component can keep a fixed pool or append qualified permanent neural
players. It does not run evaluations, update best.pt, refresh replay, or write
the transition archive. The training runner calls it after a durable evaluation
result is available and saves the resulting membership in recovery checkpoints.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from math import isfinite
from pathlib import Path
import re
from typing import Literal

from players.catalogue import (
    DEFAULT_WEIGHTS_ROOT, ModelSpec, as_model_spec, load_neural_model, snapshot_pool,
)
from players.neural.artifacts import neural_policy_id, save_neural_artifact
from players.neural.q_network import QNetwork
from players.rule_based.registry import OpponentSpec

from .config import TrainingConfig
from .episodes import EpisodeResult
from .evaluation import EvaluationResult, EvaluationSummary
from .rewards import EpisodeOutcome, TIMEOUT_REWARD


def _json(value) -> str:
    """Preserve numeric types when comparing exact, finite JSON pool records."""
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("evaluated_pool must contain finite JSON records") from error


def _reward(value, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite reward between -1.5 and 1.5")
    try:
        valid = isfinite(value) and -1.5 <= value <= 1.5
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite reward between -1.5 and 1.5")
    return float(value)


def _validate_evaluation(evaluation: EvaluationResult | EvaluationSummary, config: TrainingConfig) -> None:
    """Reject inconsistent summaries before they can qualify a model."""
    if not isinstance(evaluation, (EvaluationResult, EvaluationSummary)):
        raise ValueError("evaluation must be an EvaluationResult or EvaluationSummary")
    if type(evaluation.attempted_games) is not int or evaluation.attempted_games != config.evaluation_games:
        raise ValueError("evaluation attempted_games must match config.evaluation_games")
    if type(evaluation.capped_games) is not int or not 0 <= evaluation.capped_games <= evaluation.attempted_games:
        raise ValueError("evaluation capped_games is invalid")
    legacy = isinstance(evaluation, EvaluationSummary) and evaluation.capped_mask is None
    completed = evaluation.attempted_games - evaluation.capped_games if legacy else evaluation.attempted_games
    if not isinstance(evaluation.rewards, tuple) or len(evaluation.rewards) != completed:
        raise ValueError("evaluation rewards must match the completed-game count")
    for reward in evaluation.rewards:
        _reward(reward, "evaluation reward")
    if isinstance(evaluation, EvaluationResult):
        if not isinstance(evaluation.episodes, tuple) or len(evaluation.episodes) != completed:
            raise ValueError("evaluation episodes must match the completed-game count")
        _reward(evaluation.promotion_threshold, "promotion_threshold")
        if not evaluation.valid:
            raise ValueError("evaluation timeout counts or rewards do not match its episodes")
        for episode, reward in zip(evaluation.episodes, evaluation.rewards):
            if not isinstance(episode, EpisodeResult) or not isinstance(episode.outcome, EpisodeOutcome) or (
                _reward(episode.outcome.reward, "episode reward") != reward
            ):
                raise ValueError("evaluation rewards must match their completed episodes")
    elif not legacy:
        if not isinstance(evaluation.capped_mask, tuple) or len(evaluation.capped_mask) != completed or (
            any(type(item) is not bool for item in evaluation.capped_mask)
            or sum(evaluation.capped_mask) != evaluation.capped_games
            or any(capped and reward != TIMEOUT_REWARD for capped, reward in zip(evaluation.capped_mask, evaluation.rewards))
        ):
            raise ValueError("evaluation timeout flags do not match its rewards")
    if completed:
        mean = _reward(evaluation.mean_reward, "mean_reward")
        if mean != sum(evaluation.rewards) / completed:
            raise ValueError("evaluation mean_reward does not match its rewards")
    elif evaluation.mean_reward is not None:
        raise ValueError("mean_reward must be None when no evaluation game finished")


@dataclass(frozen=True, slots=True)
class AdmissionRecord:
    """One actual pool join; store its evaluated pool without mutable aliases.

    evaluated_pool and to_dict return new JSON containers on every call. The
    permanent artifact may have older provenance if it was exported previously;
    this record describes when that existing policy joined this particular run.
    """

    run_id: str
    completed_games: int
    model_id: str
    artifact_path: Path
    mean_reward: float
    threshold: float
    _evaluated_pool_json: str = field(repr=False)

    @property
    def evaluated_pool(self) -> list[dict]:
        return json.loads(self._evaluated_pool_json)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "completed_games": self.completed_games,
            "model_id": self.model_id, "artifact_path": str(self.artifact_path),
            "mean_reward": self.mean_reward, "threshold": self.threshold,
            "evaluated_pool": self.evaluated_pool,
        }


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Distinguish qualifying from joining: an existing member can qualify again."""

    qualified: bool
    added: bool
    reason: Literal["fixed_pool", "capped_evaluation", "below_threshold", "already_present", "admitted"]
    model_id: str | None = None
    artifact_path: Path | None = None


class OpponentPool:
    """Own ordered pool membership and append eligible frozen neural opponents.

    Sampling remains the episode collector's job: pass opponents to it as usual.
    Taking a snapshot before evaluation records the exact opponent policies used.
    consider_candidate requires that same pool to still be current. This class
    does not prove that the supplied network produced the evaluation: callers
    must evaluate a frozen copy and pass those exact, unchanged weights here.
    """

    def __init__(
        self, opponents: Sequence[OpponentSpec | ModelSpec], config: TrainingConfig,
        *, run_id: str, weights_root: str | Path = DEFAULT_WEIGHTS_ROOT,
        device: str = "cpu",
    ) -> None:
        if not isinstance(config, TrainingConfig):
            raise ValueError("config must be a TrainingConfig")
        if type(run_id) is not str or not run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if type(device) is not str or re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", device) is None:
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
        try:
            models = tuple(as_model_spec(item) for item in opponents)
        except TypeError as error:
            raise ValueError("opponents must be a nonempty sequence of model specifications") from error
        snapshot_pool(models)  # Require unique, persistable policies up front.
        self._opponents = models
        self._config = config
        self._run_id = run_id
        self._weights_root = Path(weights_root).resolve()
        self._device = device
        self._admissions: tuple[AdmissionRecord, ...] = ()

    @property
    def opponents(self) -> tuple[ModelSpec, ...]:
        """Return ordered members; every factory creates a separate controller."""
        return self._opponents

    @property
    def admissions(self) -> tuple[AdmissionRecord, ...]:
        """Return the immutable sequence of actual joins made by this instance."""
        return self._admissions

    def snapshot(self) -> list[dict]:
        """Return fresh records of current membership and exact policy settings."""
        return snapshot_pool(self._opponents)

    def consider_candidate(
        self, candidate: QNetwork, evaluation: EvaluationResult | EvaluationSummary, *,
        completed_games: int, evaluated_pool: Sequence[dict],
    ) -> AdmissionResult:
        """Append a policy only after a complete evaluation strictly beats the bar.

        No minimum gap or admission quota applies. Timeout scores participate
        in the mean; the demonstration-promotion threshold has no effect here.
        Reuse a library entry's original name and provenance. Already-present
        policies qualify without another file write, load, or membership change.

        Validate everything before mutating membership. A failed export/load
        leaves membership and records unchanged (a completed export can remain
        available in the library). The runner persists membership and records;
        recovery can supply an EvaluationSummary from its validated journal.
        """
        if not isinstance(candidate, QNetwork):
            raise ValueError("candidate must be a QNetwork")
        if type(completed_games) is not int or completed_games < 0:
            raise ValueError("completed_games must be a nonnegative integer")
        if self._admissions and completed_games < self._admissions[-1].completed_games:
            raise ValueError("completed_games cannot precede an earlier admission")
        if not isinstance(evaluated_pool, (list, tuple)):
            raise ValueError("evaluated_pool must contain the full snapshot taken before evaluation")
        evaluated_json = _json(evaluated_pool)
        if evaluated_json != _json(self.snapshot()):
            raise ValueError("evaluated_pool differs from the current pool; evaluate against the current snapshot")
        _validate_evaluation(evaluation, self._config)
        if self._config.opponent_pool_mode == "fixed":
            return AdmissionResult(False, False, "fixed_pool")
        if not evaluation.valid:
            return AdmissionResult(False, False, "capped_evaluation")
        threshold = self._config.opponent_admission_threshold
        if evaluation.mean_reward <= threshold:
            return AdmissionResult(False, False, "below_threshold")

        model_id = neural_policy_id(candidate)
        for member in self._opponents:
            if member.model_id == model_id:
                return AdmissionResult(True, False, "already_present", model_id,
                                       Path(member.config["artifact_path"]))

        path = self._weights_root / model_id
        if not path.exists():
            path = save_neural_artifact(
                candidate, root=self._weights_root,
                name=f"Neural {self._run_id} game {completed_games}",
                provenance={
                    "source": "opponent_admission", "run_id": self._run_id,
                    "completed_games": completed_games,
                    "mean_reward": evaluation.mean_reward,
                    "opponent_admission_threshold": threshold,
                    "evaluated_pool": json.loads(evaluated_json),
                },
            )
        member = load_neural_model(path, device=self._device)
        if member.model_id != model_id:
            raise ValueError("saved candidate identity differs from the evaluated weights")
        members = (*self._opponents, member)
        snapshot_pool(members)  # Check the complete new specification before joining.
        record = AdmissionRecord(
            self._run_id, completed_games, model_id, path.resolve(),
            float(evaluation.mean_reward), threshold, evaluated_json,
        )
        self._opponents = members
        self._admissions = (*self._admissions, record)
        return AdmissionResult(True, True, "admitted", model_id, path.resolve())
