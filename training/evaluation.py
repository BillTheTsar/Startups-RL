"""Evaluate one frozen online checkpoint in fresh, non-training games.

Every requested game counts, including a timeout with reward -1. This module
returns complete episode experience for the caller to archive
or add to replay; it never writes files or performs optimizer updates.
"""

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from random import Random

from players.neural import FlatObservationEncoder, NeuralPlayer
from players.neural.q_network import QNetwork
from players.catalogue import ModelSpec
from players.rule_based.registry import OpponentSpec, RULE_BASED_OPPONENTS

from .config import TrainingConfig
from .dataset import Partition
from .episodes import EpisodeResult, collect_episode, create_episode
from .rewards import EpisodeOutcome, TIMEOUT_REWARD
from .transitions import Transition


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Scores retained for admission recovery when trajectories are not archived.

    No episode data is needed to decide whether a frozen policy qualifies.
    The opponent manager validates these counts and recomputes the mean before
    using them. The runner also checks the saved model and opponent identities.
    """

    rewards: tuple[float, ...]
    attempted_games: int
    capped_games: int
    mean_reward: float | None
    capped_mask: tuple[bool, ...] | None = None

    @property
    def valid(self) -> bool:
        # None denotes a legacy saved result where capped games had no reward.
        return self.attempted_games > 0 and len(self.rewards) == self.attempted_games and (
            self.capped_games == 0 if self.capped_mask is None else
            len(self.capped_mask) == self.attempted_games and sum(self.capped_mask) == self.capped_games
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One block's completed games and its eligibility for selection/promotion.

    episodes and rewards contain every game, including timeout endings, in
    attempted-game order. Timeout rewards are -1 and never outright wins.
    The threshold is captured from this block's
    configuration so partitioned_episodes uses the same qualification rule.
    """

    episodes: tuple[EpisodeResult, ...]
    rewards: tuple[float, ...]
    attempted_games: int
    capped_games: int
    mean_reward: float | None
    promotion_threshold: float

    @property
    def valid(self) -> bool:
        """Require one correctly classified result for every requested game."""
        return (
            self.attempted_games > 0
            and len(self.episodes) == len(self.rewards) == self.attempted_games
            and all(isinstance(episode, EpisodeResult) and isinstance(episode.outcome, EpisodeOutcome)
                    and type(episode.capped) is bool for episode in self.episodes)
            and sum(episode.capped for episode in self.episodes) == self.capped_games
            and all(episode.outcome.reward == reward and (
                not episode.capped or reward == TIMEOUT_REWARD and not episode.outcome.outright_win
            ) for episode, reward in zip(self.episodes, self.rewards))
        )

    @property
    def qualifies(self) -> bool:
        """Require a valid block with mean strictly above its promotion threshold."""
        return self.valid and self.mean_reward is not None and self.mean_reward > self.promotion_threshold

    def partitioned_episodes(self) -> tuple[tuple[Partition, tuple[Transition, ...]], ...]:
        """Return complete trajectories for a qualifying block, otherwise nothing.

        Only outright wins become demo. Tied first and every other outcome are
        non_demo. Keep both partitions' games in their original relative order;
        callers can append this sequence to live replay and the archive once.
        """
        if not self.qualifies:
            return ()
        return tuple(
            ("demo" if episode.outcome.outright_win else "non_demo", episode.transitions)
            for episode in self.episodes if episode.transitions
        )


async def evaluate_checkpoint(
    online: QNetwork,
    config: TrainingConfig,
    *,
    rng: Random,
    opponents: Sequence[OpponentSpec | ModelSpec] = RULE_BASED_OPPONENTS,
) -> EvaluationResult:
    """Freeze online once, then attempt exactly config.evaluation_games games.

    Validate arguments before cloning or consuming the supplied evaluation RNG.
    The independent copy retains the original device/dtype, enters evaluation
    mode and has gradients disabled/cleared. Original weights, modes, gradients
    and training RNGs are untouched. The original network may change while this
    coroutine awaits; all games still use the copy taken at the block's start.

    Each fresh game uses a greedy NeuralPlayer (epsilon zero), independent
    uniform seat/starter choices and three opponent draws with replacement.
    Reuse create_episode and collect_episode without a training callback.
    Advance only the supplied evaluation RNG and fresh game/controller RNGs.

    Record focal trajectories for natural and timeout endings. Timeouts count
    toward the mean, not as replacement attempts or invalid blocks. Actual
    failures or cancellation propagate. No replay, optimizer, target network, checkpoint,
    archive, or training counter is owned or modified here.
    """
    if not isinstance(online, QNetwork):
        raise ValueError("online must be a QNetwork")
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    if not isinstance(rng, Random):
        raise ValueError("rng must be an evaluation Random instance")
    try:
        opponents = tuple(opponents)
    except TypeError as error:
        raise ValueError("opponents must be a nonempty sequence of OpponentSpec or ModelSpec entries") from error
    if not opponents or any(not isinstance(opponent, (OpponentSpec, ModelSpec)) for opponent in opponents):
        raise ValueError("opponents must be a nonempty sequence of OpponentSpec or ModelSpec entries")

    frozen = deepcopy(online).eval().requires_grad_(False)
    frozen.zero_grad(set_to_none=True)
    encoder = FlatObservationEncoder()

    def focal_factory(player_id, *, seed):
        return NeuralPlayer(
            player_id, frozen.predict, encoder, name="Evaluation checkpoint",
            selection_mode="greedy", epsilon=0.0, seed=seed,
        )

    completed = []
    capped = 0
    for _ in range(config.evaluation_games):
        # Most game/controller awaits complete synchronously. Yield between
        # attempts so cancellation does not wait for an entire evaluation block.
        await asyncio.sleep(0)
        episode = create_episode(focal_factory, rng=rng, opponents=opponents)
        result = await collect_episode(episode, max_actions=config.max_actions_per_game)
        capped += int(result.capped)
        completed.append(result)

    rewards = tuple(episode.outcome.reward for episode in completed)
    return EvaluationResult(
        episodes=tuple(completed), rewards=rewards,
        attempted_games=config.evaluation_games, capped_games=capped,
        mean_reward=sum(rewards) / len(rewards) if rewards else None,
        promotion_threshold=config.promotion_threshold,
    )
