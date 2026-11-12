"""Translate normal settlement or an action-limit ending into learning rewards."""

from collections.abc import Mapping
from dataclasses import dataclass

from evaluation import TIMEOUT_POINTS as TIMEOUT_REWARD, score_placements
from simulator import PlayerId


REWARD_SCHEME = "placement_points_with_timeout_v2"


@dataclass(frozen=True)
class EpisodeOutcome:
    """Learning reward and outright-win status for a normal or capped ending.

    Normal endings use placement points. Every seat receives TIMEOUT_REWARD at
    an unfinished action-limit ending, and none counts as an outright winner.
    These rewards are separate from the simulator's integer coin balances.
    """

    reward: float
    outright_win: bool


def get_timeout_outcome() -> EpisodeOutcome:
    """Return the same -1 reward and no outright win for any timed-out seat."""
    return EpisodeOutcome(TIMEOUT_REWARD, False)


def get_episode_outcome(scores: Mapping[PlayerId, int], player_id: PlayerId) -> EpisodeOutcome:
    """Award 1.5, 0.5, -0.5, -1.5, averaging occupied places for any ties.

    Require all four final balances keyed by seat ID. Tied first is not an
    outright win. Collection code separately decides demonstration eligibility;
    this function does not select a partition. Reuse the evaluation module's
    placement rules rather than introducing a second rule.
    """
    if type(player_id) is not int or player_id not in (1, 2, 3, 4):
        raise ValueError("player_id must be an integer from 1 through 4")
    if not isinstance(scores, Mapping) or (
        any(type(pid) is not int for pid in scores) or set(scores) != {1, 2, 3, 4}
    ):
        raise ValueError("scores must map all four seat IDs to their final balances")
    placement = score_placements(tuple(scores[pid] for pid in (1, 2, 3, 4)))[player_id - 1]
    return EpisodeOutcome(placement.points, placement.outright_win)
