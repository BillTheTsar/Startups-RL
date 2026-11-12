"""Seeded four-player evaluation, independent of presentation and persistence.

Sample four distinct model IDs uniformly into seats, then sample the starter
uniformly. Award 1.5, 0.5, -0.5, -1.5 points, averaging occupied places on ties.
At the action limit, an unfinished game awards -1 to every player, with no
placement or pairwise result. Its actual coin balances are not settled.
Only normalized head-to-head results are exposed; encounter counts and net
results are internal sufficient statistics. No files are written here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from secrets import randbits
import sys
from typing import TYPE_CHECKING

from action_codec import ActionCodec
from simulator import PlayerId, Simulator

if TYPE_CHECKING:
    from game_loop import GameStep
    from players.base import Player


EVALUATION_VERSION = 2
PLACEMENT_POINTS = (1.5, 0.5, -0.5, -1.5)
TIMEOUT_POINTS = -1.0
DEFAULT_MAX_ACTIONS = 10_000


def __getattr__(name):
    """Preserve old shared-type imports without loading players for report rebuilds."""
    if name in ("PlayerFactory", "ModelSpec"):
        from players import catalogue
        return getattr(catalogue, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _copy_config(config):
    from players.catalogue import _copy_config as copy_config
    return copy_config(config)


def _weight_json(value):
    from players.catalogue import _weight_json as weight_json
    return weight_json(value)


def builtin_models() -> tuple[ModelSpec, ...]:
    """Load the shared rule-only catalogue only when players are needed."""
    from players.catalogue import builtin_models as load_builtin_models
    return load_builtin_models()


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_models(models: Sequence[ModelSpec]) -> tuple[ModelSpec, ...]:
    from players.catalogue import ModelSpec

    models = tuple(models)
    if len(models) < 4 or any(not isinstance(model, ModelSpec) for model in models):
        raise ValueError("select at least four ModelSpec entries")
    if len({model.model_id for model in models}) != len(models):
        raise ValueError("model IDs must be unique; repeated models are not allowed")
    return models


@dataclass(frozen=True)
class GamePlan:
    """Complete setup metadata. Model IDs and player seeds are in seat 1..4 order."""

    game_number: int
    game_seed: int
    model_ids: tuple[str, ...]
    starting_player_id: PlayerId
    deck_seed: int
    player_seeds: tuple[int, ...]


def make_game_plan(model_ids: Sequence[str], game_number: int, game_seed: int) -> GamePlan:
    """Uniform ordered sampling without replacement; no balancing or reweighting."""
    if len(model_ids) < 4 or len(set(model_ids)) != len(model_ids):
        raise ValueError("the model pool must contain at least four distinct IDs")
    _positive_integer(game_number, "game_number")
    if type(game_seed) is not int:
        raise ValueError("game_seed must be an integer")
    rng = Random(game_seed)
    lineup = tuple(rng.sample(tuple(model_ids), 4))
    starter = rng.choice(Simulator.PLAYER_IDS)
    deck_seed = rng.getrandbits(64)
    player_seeds = tuple(rng.getrandbits(64) for _ in Simulator.PLAYER_IDS)
    return GamePlan(game_number, game_seed, lineup, starter, deck_seed, player_seeds)


@dataclass(frozen=True)
class GameRecord:
    """Compact completed outcome, sufficient to recompute every current statistic.

    Scores are simulator coin balances in seat 1..4 order: final balances for
    normal finishes, or the last observed balances when capped is True. Timeout
    points are separate from these unsettled balances. Records are sent
    to on_game and then discarded by the runner, rather than retaining histories.
    dataclasses.asdict(record) is JSON-compatible for archive storage.
    """

    plan: GamePlan
    scores: tuple[int, ...]
    action_count: int
    capped: bool = False


@dataclass(frozen=True)
class Placement:
    points: float
    outright_win: bool
    credit: tuple[Fraction, ...]


def score_placements(scores: Sequence[int]) -> tuple[Placement, ...]:
    """Average occupied points and split placement credit equally across ties."""
    if len(scores) != 4 or any(type(score) is not int for score in scores):
        raise ValueError("scores must contain exactly four integer final balances")
    placements = []
    for score in scores:
        above = sum(other > score for other in scores)
        tied = sum(other == score for other in scores)
        credit = tuple(
            Fraction(1, tied) if above <= place < above + tied else Fraction(0)
            for place in range(4)
        )
        points = sum(PLACEMENT_POINTS[above:above + tied]) / tied
        placements.append(Placement(points, above == 0 and tied == 1, credit))
    return tuple(placements)


@dataclass(frozen=True)
class Statistics:
    appearances: int
    total_points: float
    outright_wins: int
    placement_counts: tuple[float, ...]
    capped_games: int = 0

    @property
    def mean_points(self) -> float | None:
        return self.total_points / self.appearances if self.appearances else None

    @property
    def outright_win_rate(self) -> float | None:
        return self.outright_wins / self.appearances if self.appearances else None

    @property
    def placement_distribution(self) -> tuple[float | None, ...]:
        """Fractions of all appearances; caps have no credit in any placement."""
        return tuple(count / self.appearances if self.appearances else None for count in self.placement_counts)

    @property
    def capped_rate(self) -> float | None:
        return self.capped_games / self.appearances if self.appearances else None


@dataclass(frozen=True)
class ModelStatistics:
    model_id: str
    overall: Statistics
    by_position: tuple[Statistics, ...]  # First, second, third, fourth to act.


@dataclass(frozen=True)
class EvaluationResult:
    manifest: dict[str, object]
    completed_games: int
    models: tuple[ModelStatistics, ...]
    head_to_head: tuple[tuple[float | None, ...], ...]
    capped_games: int = 0


@dataclass
class _Totals:
    appearances: int = 0
    points: float = 0.0
    outright_wins: int = 0
    placements: list[Fraction] = field(default_factory=lambda: [Fraction(0) for _ in range(4)])
    capped_games: int = 0

    def add(self, placement: Placement, *, capped: bool = False) -> None:
        self.appearances += 1
        self.points += placement.points
        self.outright_wins += placement.outright_win
        self.capped_games += capped
        for index, credit in enumerate(placement.credit):
            self.placements[index] += credit

    def snapshot(self) -> Statistics:
        return Statistics(self.appearances, self.points, self.outright_wins,
                          tuple(map(float, self.placements)), self.capped_games)


class EvaluationAccumulator:
    """Aggregate compact records with O(m squared) memory, independent of game count."""

    def __init__(self, model_ids: Sequence[str]) -> None:
        if len(model_ids) < 4 or len(set(model_ids)) != len(model_ids):
            raise ValueError("the model pool must contain at least four distinct IDs")
        self.model_ids = tuple(model_ids)
        self._indices = {model_id: index for index, model_id in enumerate(model_ids)}
        self._overall = [_Totals() for _ in model_ids]
        self._positions = [[_Totals() for _ in range(4)] for _ in model_ids]
        self._pairs: dict[tuple[int, int], tuple[int, int]] = {}
        self.completed_games = 0
        self.capped_games = 0

    def add(self, record: GameRecord) -> None:
        """Validate the complete outcome before changing any aggregate."""
        plan = record.plan
        if len(plan.model_ids) != 4 or len(set(plan.model_ids)) != 4:
            raise ValueError("each game requires four distinct models")
        if any(model_id not in self._indices for model_id in plan.model_ids):
            raise ValueError("game contains an unknown model ID")
        if type(plan.starting_player_id) is not int or plan.starting_player_id not in Simulator.PLAYER_IDS:
            raise ValueError("starting_player_id must be an integer from 1 through 4")
        if type(record.capped) is not bool:
            raise ValueError("capped must be a boolean")
        placements = score_placements(record.scores)  # Validate diagnostic balances too.
        if record.capped:
            placements = (Placement(TIMEOUT_POINTS, False, (Fraction(0),) * 4),) * 4
        indices = tuple(self._indices[model_id] for model_id in plan.model_ids)
        for seat, (index, placement) in enumerate(zip(indices, placements), start=1):
            self._overall[index].add(placement, capped=record.capped)
            self._positions[index][(seat - plan.starting_player_id) % 4].add(placement, capped=record.capped)
        # An unfinished game's balances do not establish pairwise winners.
        for left in range(0 if record.capped else 4):
            for right in range(left + 1, 4):
                a, b = indices[left], indices[right]
                outcome = (record.scores[left] > record.scores[right]) - (record.scores[left] < record.scores[right])
                if a > b:
                    a, b, outcome = b, a, -outcome
                net, encounters = self._pairs.get((a, b), (0, 0))
                self._pairs[a, b] = net + outcome, encounters + 1
        self.completed_games += 1
        self.capped_games += record.capped

    def result(self, manifest: dict[str, object]) -> EvaluationResult:
        size = len(self.model_ids)
        matrix = [[0.0 if row == column else None for column in range(size)] for row in range(size)]
        for (a, b), (net, encounters) in self._pairs.items():
            matrix[a][b] = net / encounters
            matrix[b][a] = -net / encounters if net else 0.0
        models = tuple(
            ModelStatistics(model_id, self._overall[index].snapshot(),
                            tuple(position.snapshot() for position in self._positions[index]))
            for index, model_id in enumerate(self.model_ids)
        )
        return EvaluationResult(manifest, self.completed_games, models, tuple(map(tuple, matrix)), self.capped_games)


def create_manifest(models: Sequence[ModelSpec], num_games: int, seed: int, max_actions: int) -> dict[str, object]:
    """Snapshot settings and local source hashes; return data without writing files."""
    root = Path(__file__).resolve().parent
    sources = sorted((*root.glob("*.py"), *(root / "players").rglob("*.py")))
    return {
        "evaluation_version": EVALUATION_VERSION,
        "action_codec_version": ActionCodec.VERSION,
        "python_version": sys.version.split()[0],
        "seed": seed,
        "num_games": num_games,
        "players_per_game": 4,
        "max_actions_per_game": max_actions,
        "placement_points": list(PLACEMENT_POINTS),
        "tie_points": "average occupied places",
        "tie_placements": "split credit equally across occupied places",
        "timeout_points": TIMEOUT_POINTS,
        "timeout_placements": "none; rates use all appearances",
        "timeout_head_to_head": "excluded; no settled pairwise result",
        "sampling": "uniform ordered sample of four distinct model IDs; independent uniform starter",
        "position_order": "first, second, third, fourth to act relative to starter",
        "head_to_head": "(wins - losses) / encounters; ties included; unplayed=null; diagonal=0",
        "failure_policy": "stop; report failing game setup and completed-game statistics",
        "models": [{"model_id": model.model_id, "kind": model.kind, "config": _copy_config(model.config)}
                   for model in models],
        "source_sha256": {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in sources},
    }


class EvaluationGameError(RuntimeError):
    """A failed game with its setup and the results from preceding completed games."""

    def __init__(self, plan: GamePlan, partial_result: EvaluationResult, error: Exception) -> None:
        self.plan = plan
        self.partial_result = partial_result
        super().__init__(
            f"Game {plan.game_number} failed: {type(error).__name__}: {error}. "
            f"Run seed={partial_result.manifest['seed']}; game seed={plan.game_seed}; "
            f"deck seed={plan.deck_seed}; player seeds={plan.player_seeds}; "
            f"seats={plan.model_ids}; starter=P{plan.starting_player_id}."
        )


class _ActionLimitReached(Exception):
    """Private game-loop stop signal, handled as a scored timeout by evaluate."""


async def evaluate(
    models: Sequence[ModelSpec], num_games: int, *, seed: int | None = None,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    on_game: Callable[[GameRecord], None] | None = None,
    on_start: Callable[[dict[str, object]], None] | None = None,
    check_status: Callable[[], None] | None = None,
) -> EvaluationResult:
    """Run games sequentially through game_loop, with fresh players and simulators.

    A master RNG generates game seeds; each game has separate setup, deck, and
    player streams. Reset is owned by game_loop and does not reseed controllers.
    A genuinely unfinished game at the cap counts as an outcome worth -1 per
    player. It has no winner or placement. Normal settlement on the exact final
    allowed action takes precedence. Other errors stop the run without counting
    the failed game. Cancellation propagates.
    The synchronous on_game callback runs once after each result is aggregated;
    callback errors propagate directly. on_start receives the manifest before
    any game is constructed. check_status runs before each game and after each
    action so a background storage error can stop play promptly. The runner
    retains no per-game records. Player modules are loaded only when playing,
    allowing reports to be rebuilt without importing policy implementations.
    """
    from game_loop import run_game
    from players.base import Player

    models = validate_models(models)
    _positive_integer(num_games, "num_games")
    _positive_integer(max_actions, "max_actions")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer or None")
    for name, callback in (("on_game", on_game), ("on_start", on_start), ("check_status", check_status)):
        if callback is not None and not callable(callback):
            raise ValueError(f"{name} must be callable or None")
    seed = randbits(64) if seed is None else seed
    manifest = create_manifest(models, num_games, seed, max_actions)
    if on_start is not None:
        on_start(manifest)
    specifications = {model.model_id: model for model in models}
    totals = EvaluationAccumulator(tuple(specifications))
    rng = Random(seed)
    for number in range(1, num_games + 1):
        if check_status is not None:
            check_status()
        plan = make_game_plan(totals.model_ids, number, rng.getrandbits(64))
        actions = 0

        async def observe(step: GameStep) -> None:
            nonlocal actions
            actions += 1
            if check_status is not None:
                check_status()
            if actions >= max_actions and step.after.phase != "finished":
                raise _ActionLimitReached()

        try:
            simulator = Simulator(seed=plan.deck_seed, starting_player_id=plan.starting_player_id)
            players = []
            for player_id, model_id, player_seed in zip(Simulator.PLAYER_IDS, plan.model_ids, plan.player_seeds):
                spec = specifications[model_id]
                player = spec.factory(player_id, seed=player_seed)
                if (not isinstance(player, Player) or type(player.player_id) is not int
                        or player.player_id != player_id or player.kind != spec.kind):
                    raise ValueError(f"factory for {model_id!r} returned a player with the wrong type, seat, or kind")
                players.append(player)
            try:
                game_result = await run_game(simulator, players, on_step=observe)
            except _ActionLimitReached:
                # Keep public balances for diagnostics. Do not reveal hands,
                # settle companies, or pretend the Simulator finished normally.
                view = simulator.get_observation(1)
                balances = {player.player_id: player.coins for player in view.players}
                record = GameRecord(plan, tuple(balances[pid] for pid in Simulator.PLAYER_IDS),
                                    actions, capped=True)
            else:
                scores = dict(game_result.scores)
                record = GameRecord(plan, tuple(scores[pid] for pid in Simulator.PLAYER_IDS), actions)
            totals.add(record)
        except Exception as error:
            raise EvaluationGameError(plan, totals.result(manifest), error) from error
        if on_game is not None:
            on_game(record)
        # Built-in synchronous policies otherwise never yield to the event loop.
        await asyncio.sleep(0)
    return totals.result(manifest)
