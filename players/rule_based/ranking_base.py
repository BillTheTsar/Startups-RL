"""Inspect legal-action scores and sample uniformly among exact best-score ties.

RankingPlayer owns selection and optional ranking inspection. Scores use Fraction
so half-coin values and expected draw values can tie exactly. Policy compares
numeric action values directly; it never calls rank_actions or an explanation
method. Subclasses can share feature arithmetic between numeric and named outputs.
Inspection is pure; only policy selection consumes the private RNG.
"""

from abc import abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from random import Random

from action_codec import ActionCodec
from simulator import Action, Observation, PlayerId

from ..base import ActionMask
from .base import RuleBasedPlayer


ScoreContributions = tuple[tuple[str, Fraction], ...]


@dataclass(frozen=True)
class ScoredAction:
    """An immutable action and its additive contributions, including zero terms.

    rank_actions returns selection-score components. A strategy's explicit
    explain_actions may instead return gains relative to the current position;
    that method documents its score convention. For an uncertain draw, each
    contribution is its probability-weighted mean. Their sum is score. A
    duplicate market slot remains a distinct action. Normal selection does not
    allocate these inspection records.
    """

    action: Action
    contributions: ScoreContributions

    @property
    def action_id(self) -> int:
        """Return the fixed model-facing ActionCodec ID for this action."""
        return ActionCodec.encode(self.action)

    @property
    def score(self) -> Fraction:
        """Sum all contributions exactly, without rounding for display."""
        return sum((value for _, value in self.contributions), Fraction(0))


class RankingPlayer(RuleBasedPlayer):
    """Rank legal actions with a subclass evaluator and randomize only best ties.

    Retain only identity and a private RNG; do not store observations or a
    simulator. Inherit choose_action's validation wrapper. Subclasses may add
    immutable weights and must provide _score_action, without consuming RNG.
    Override _action_value for numeric evaluation without named breakdowns;
    WeightedRankingPlayer does so. Policy never builds or sorts inspection records.
    """

    def __init__(
        self, player_id: PlayerId, name: str = "Ranking player", *, seed: int | None = None
    ) -> None:
        """Create a private RNG; accept only an integer seed or None, not bool."""
        super().__init__(player_id, name)
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        self._rng = Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Preserve weights/identity; continue RNG unless explicitly reseeded.

        There is no observation history or score cache to clear. Validate first
        so an invalid seed cannot change the existing sequence.
        """
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        super().reset(seed)
        if seed is not None:
            self._rng.seed(seed)

    def rank_actions(self, observation: Observation, mask: ActionMask) -> tuple[ScoredAction, ...]:
        """Return all enabled actions sorted by descending score, then codec ID.

        Validate the request; trust the matching simulator mask for legality.
        Preserve separate market slots. Include negative scores; there is no
        additional rejection threshold. Stable display ordering does not break
        selection ties. Never consume randomness, retain views, or mutate inputs.
        This is an explicit inspection method, not part of normal selection.
        """
        self._validate_request(observation, mask)
        ranked = []
        for action_id, allowed in enumerate(mask):
            if allowed:
                action = ActionCodec.decode(action_id)
                ranked.append(ScoredAction(action, self._score_action(observation, action)))
        return tuple(sorted(ranked, key=lambda entry: (-entry.score, entry.action_id)))

    def policy(self, observation: Observation, mask: ActionMask) -> int:
        """Sample uniformly from exact top-score ties, returning a legal codec ID.

        Evaluate numeric scores once per legal action in codec order and retain
        only the best IDs. Do not call rank_actions or build/sort score reports.
        Use the private RNG only after evaluation succeeds. A unique winner
        needs no random draw. Lower-scoring legal actions are never sampled.
        Codec order preserves seeded tie choices relative to rank_actions.
        """
        self._validate_request(observation, mask)
        best: list[int] = []
        highest: Fraction | None = None
        for action_id, allowed in enumerate(mask):
            if allowed:
                value = self._action_value(observation, ActionCodec.decode(action_id))
                if highest is None or value > highest:
                    highest, best = value, [action_id]
                elif value == highest:
                    best.append(action_id)
        return best[0] if len(best) == 1 else self._rng.choice(tuple(best))

    def _action_value(self, observation: Observation, action: Action) -> Fraction:
        """Return the selection value; default to summing the subclass's features.

        This fallback supports subclasses that implement only _score_action.
        Override it to avoid creating named contributions during gameplay, using
        the same underlying arithmetic as _score_action. WeightedRankingPlayer's
        override computes no baseline, labels, or explanation records.
        """
        return sum((value for _, value in self._score_action(observation, action)), Fraction(0))

    @abstractmethod
    def _score_action(self, observation: Observation, action: Action) -> ScoreContributions:
        """Score one enabled action using only the supplied view, without mutation.

        Called by rank_actions after request validation, once per enabled action;
        also used by _action_value's fallback if a subclass does not override it.
        Return immutable named Fraction contributions, whose sum ranks the action.
        Subclasses define uncertainty estimates and weights. Do not query hidden
        simulator state, sample hypothetical cards, or consume the player's RNG.
        """
        raise NotImplementedError
