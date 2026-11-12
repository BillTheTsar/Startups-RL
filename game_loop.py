"""Run four Player controllers against a Simulator without owning game rules.

The optional observer receives only one seat's observations and public actions.
Players receive their own views after initialization and every successful step,
including the terminal step. No complete SimulatorState is passed to a player.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from action_codec import ActionCodec
from players.base import Player
from simulator import Action, Observation, PlayerId, Simulator


@dataclass(frozen=True)
class GameStep:
    """A completed action and its before/after views for the selected observer."""

    player_id: PlayerId
    action: Action
    before: Observation
    after: Observation


@dataclass(frozen=True)
class GameResult:
    """Official results plus observer views needed to explain final settlement.

    The final play action does not move coins, so its preceding balances are
    also the balances immediately before settlement. Final investments include
    all revealed hands. Hidden removed cards are absent from these records.
    """

    before_final_action: Observation
    final_observation: Observation
    scores: tuple[tuple[PlayerId, int], ...]
    winner_ids: tuple[PlayerId, ...]


StepObserver = Callable[[GameStep], Awaitable[None]]


async def run_game(
    simulator: Simulator,
    players: Sequence[Player],
    *,
    viewer_id: PlayerId = 1,
    on_step: StepObserver | None = None,
) -> GameResult:
    """Reset four controllers, notify them, then await and execute their decisions.

    Require exactly one Player per seat and a live simulator. Call reset once
    without reseeding; the caller can construct seeded stochastic controllers.
    Deliver the supplied game's current view once before play and a new view
    once after every action. Historical observations are not replayed here.

    Await the observer after each successful step and all player notifications;
    a CLI may pause there. Do not retry bad decisions or swallow policy errors,
    cancellation, or EOF. A rejected or cancelled decision never advances the
    game. On success, return official scores and the final observer snapshots.
    """
    if type(viewer_id) is not int or viewer_id not in Simulator.PLAYER_IDS:
        raise ValueError("viewer_id must be an integer from 1 through 4")
    if len(players) != 4 or any(not isinstance(player, Player) for player in players):
        raise ValueError("players must contain one Player for each of the four seats")
    if any(type(player.player_id) is not int for player in players):
        raise ValueError("Player seat IDs must be integers")
    seats = {player.player_id: player for player in players}
    if set(seats) != set(Simulator.PLAYER_IDS):
        raise ValueError("Player seat IDs must be distinct and cover 1 through 4")
    if simulator.is_terminal:
        raise ValueError("Cannot run an already finished game")

    def notify_players() -> dict[PlayerId, Observation]:
        views = {pid: simulator.get_observation(pid) for pid in Simulator.PLAYER_IDS}
        for pid in Simulator.PLAYER_IDS:
            seats[pid].on_observation(views[pid])
        return views

    for pid in Simulator.PLAYER_IDS:
        seats[pid].reset()
    views = notify_players()
    while not simulator.is_terminal:
        player_id = simulator.current_player_id
        before = views[viewer_id]
        mask = ActionCodec.make_mask(simulator.get_legal_actions(player_id))
        action_id = await seats[player_id].choose_action(views[player_id], mask)
        action = ActionCodec.decode(action_id)
        simulator.step(player_id, action)
        views = notify_players()
        if on_step is not None:
            await on_step(GameStep(player_id, action, before, views[viewer_id]))

    return GameResult(
        before_final_action=before,
        final_observation=views[viewer_id],
        scores=tuple(simulator.get_scores().items()),
        winner_ids=simulator.get_winner_ids(),
    )
