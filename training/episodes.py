"""Collect one focal player's decisions through the existing game loop."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from random import Random

from evaluation import DEFAULT_MAX_ACTIONS, PlayerFactory
from game_loop import GameResult, GameStep, run_game
from players.base import ActionMask, Player, PlayerKind
from players.catalogue import ModelSpec
from players.rule_based.registry import OpponentSpec, RULE_BASED_OPPONENTS
from simulator import Observation, PlayerId, Simulator

from .rewards import EpisodeOutcome, get_episode_outcome, get_timeout_outcome
from .transitions import Transition, TransitionBuilder


TransitionCallback = Callable[[Transition], Awaitable[None]]


@dataclass(frozen=True)
class Episode:
    """A fresh game and four controllers, with exactly one designated collector."""

    simulator: Simulator
    players: tuple[Player, ...]
    focal_player_id: PlayerId


@dataclass(frozen=True)
class EpisodeResult:
    """A normally settled or capped game's complete focal training episode.

    A capped ending has reward -1, capped=True and game_result=None because no
    coin settlement occurred. final_observation is the player's actual last
    live view; its phase is not rewritten to finished. Normal endings retain
    their official GameResult. A focal seat that never acted has no transitions.
    """

    transitions: tuple[Transition, ...]
    outcome: EpisodeOutcome
    game_result: GameResult | None
    action_count: int
    capped: bool = False
    final_observation: Observation | None = None


class EpisodeLimitExceeded(RuntimeError):
    """Legacy cap exception retained for callers importing the old interface.

    collect_episode now returns an explicit capped result instead. Exceptions
    raised by a controller or callback still propagate, including this class.
    """


class _EpisodeCapReached(RuntimeError):
    """Stop the shared game loop after its final allowed live observation."""

    def __init__(self, observation: Observation) -> None:
        self.observation = observation


def create_episode(
    focal_factory: PlayerFactory, *, rng: Random | None = None,
    opponents: Sequence[OpponentSpec | ModelSpec] = RULE_BASED_OPPONENTS,
) -> Episode:
    """Randomize the focal seat and starter independently, then sample opponents.

    Draw three entries independently and uniformly WITH replacement. Construct
    fresh controllers with separate seeds even when a strategy repeats. The
    focal factory accepts (player_id, *, seed). Opponents may be rule-based
    specifications or a fixed mixed catalogue. Neural factories share frozen
    weights while creating separate controllers. No human UI is involved.
    """
    try:
        opponents = tuple(opponents)
    except TypeError as error:
        raise ValueError("opponents must contain OpponentSpec or ModelSpec entries") from error
    if not opponents or any(not isinstance(item, (OpponentSpec, ModelSpec)) for item in opponents):
        raise ValueError("opponents must contain a nonempty sequence of OpponentSpec or ModelSpec entries")
    rng = Random() if rng is None else rng
    focal_id = rng.choice(Simulator.PLAYER_IDS)
    starting_id = rng.choice(Simulator.PLAYER_IDS)
    sampled = iter(rng.choice(opponents) for _ in range(3))
    # Resolve all opponent draws before consuming controller/deck seeds.
    sampled = iter(tuple(sampled))
    simulator = Simulator(seed=rng.getrandbits(64), starting_player_id=starting_id)
    players = tuple(
        focal_factory(pid, seed=rng.getrandbits(64)) if pid == focal_id
        else next(sampled).create(pid, seed=rng.getrandbits(64))
        for pid in Simulator.PLAYER_IDS
    )
    if any(not isinstance(player, Player) or player.player_id != pid for pid, player in zip(Simulator.PLAYER_IDS, players)):
        raise ValueError("factories must return a fresh Player for the requested seat")
    return Episode(simulator, players, focal_id)


class _RecordingPlayer(Player):
    """Delegate controller lifecycle and decisions, recording only this seat."""

    def __init__(self, player: Player, on_transition: TransitionCallback | None = None) -> None:
        super().__init__(player.player_id, player.name)
        self.player = player
        self.builder = TransitionBuilder(player.player_id)
        self.transitions: list[Transition] = []
        self.on_transition = on_transition

    @property
    def kind(self) -> PlayerKind:
        return self.player.kind

    def reset(self, seed: int | None = None) -> None:
        self.builder.reset()
        self.transitions.clear()
        self.player.reset(seed)

    def on_observation(self, observation: Observation) -> None:
        self.player.on_observation(observation)

    async def choose_action(self, observation: Observation, mask: ActionMask) -> int:
        self._validate_request(observation, mask)
        completed = self.builder.complete_pending(observation, mask)
        if completed is not None:
            await self.record_transition(completed)
        action_id = self._validate_action_id(await self.player.choose_action(observation, mask), mask)
        self.builder.begin_action(observation, action_id, mask)
        return action_id

    async def record_transition(self, transition: Transition) -> None:
        """Stage this seat's completed transition, then await the optional hook."""
        self.transitions.append(transition)
        if self.on_transition is not None:
            await self.on_transition(transition)


async def collect_episode(
    episode: Episode, *, max_actions: int = DEFAULT_MAX_ACTIONS,
    on_transition: TransitionCallback | None = None,
) -> EpisodeResult:
    """Finish normally or at the action cap, then close the focal trajectory.

    Reuse run_game's masks, observations, callbacks, and Simulator validation.
    Other seats may use the identical strategy: only focal_player_id is recorded.
    At the cap, accept a genuinely terminal step with its normal outcome.
    Otherwise end this training episode with reward -1 and no outright winner.
    The live simulator is not settled, its coin balances stay unchanged, and
    players do not learn opponents' hidden hands. Return capped=True and the
    final actual view; every recorded action still has one complete transition.

    An optional callable returning an awaitable receives each completed focal
    transition exactly once. Await intermediate callbacks before
    the next focal action is selected, so that choice can use updated weights.
    Await the terminal callback after normal settlement or a timeout ending.
    No callback runs for the initial decision or an incomplete pending action;
    callbacks do not request additional observations or run for other seats.

    Callbacks may train from existing replay while this game's transitions stay
    staged until completion. A failed game does not roll back earlier callbacks.
    This function does not append replay or archive entries. Policy
    and callback errors propagate rather than becoming discarded-game results.
    Reject noncallable hooks before play. A callable that violates its awaitable
    return contract raises when its first completed transition is delivered.
    """
    if type(max_actions) is not int or max_actions <= 0:
        raise ValueError("max_actions must be a positive integer")
    if on_transition is not None and not callable(on_transition):
        raise ValueError("on_transition must be callable or None")
    if type(episode.focal_player_id) is not int or episode.focal_player_id not in Simulator.PLAYER_IDS:
        raise ValueError("focal_player_id must be an integer from 1 through 4")
    focal = [player for player in episode.players if player.player_id == episode.focal_player_id]
    if len(focal) != 1:
        raise ValueError("episode must have exactly one focal player")
    recorder = _RecordingPlayer(focal[0], on_transition)
    players = tuple(recorder if player is focal[0] else player for player in episode.players)
    action_count = 0

    async def count_step(step: GameStep) -> None:
        nonlocal action_count
        action_count += 1
        if action_count >= max_actions and step.after.phase != "finished":
            raise _EpisodeCapReached(step.after)

    try:
        result = await run_game(episode.simulator, players, viewer_id=episode.focal_player_id, on_step=count_step)
    except _EpisodeCapReached as ending:
        result = None
        capped = True
        final_observation = ending.observation
        outcome = get_timeout_outcome()
        terminal = recorder.builder.finish_timeout(final_observation, outcome.reward)
    else:
        capped = False
        final_observation = result.final_observation
        outcome = get_episode_outcome(dict(result.scores), episode.focal_player_id)
        terminal = recorder.builder.finish_episode(final_observation, outcome.reward)
    if terminal is not None:
        await recorder.record_transition(terminal)
    return EpisodeResult(tuple(recorder.transitions), outcome, result, action_count,
                         capped=capped, final_observation=final_observation)
