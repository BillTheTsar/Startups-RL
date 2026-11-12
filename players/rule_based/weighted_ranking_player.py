"""Immediate resulting-position scoring with tunable additive action events.

SELECTION SCORE = position(after) + action-event adjustments.
The current position value is constant across actions and is not computed for
selection. Take evaluates the proposed four-card hand; play evaluates the
resulting three-card hand. Investing preserves combined holdings; offering reduces them.
Score only the immediate action, even on the last turn: no compulsory-play
lookahead, opponent pickup simulation, or endgame-specific weights.

Position components (sum per company where appropriate):
* hand_cards: +1 per 5/6, 0 per 7/8, -1 per 9/10, only cards in hand.
* hand_groups: 0 for 0/1 copies, +1 for 2, +3 for 3, +5 for 4; not cumulative.
* coins: balance / 2, including negative balances, without floor division.
* excess_holdings: -1 per hand+investment card above (company_id + 1) // 2.
* majority_position: compare own hand+investments against the largest opponent
  investment+known-hand count: gap <= -2: -4; -1: -2; 0: 0; 1: +1; >= 2: +3.
  A company with no owned cards contributes 0. All opponents count here.
Selection uses the resulting POSITION values directly. Explicit explain_actions
inspection subtracts each current-position component once per request to report
changes: losing a bonus is negative; removing a penalty is positive. This shifts
all scores equally, preserving their order and ties.

Action-event components (zero outside the specified action):
* pickup_commitment: +1 per existing own investment in the acquired company,
  on both market pickups and possible deck outcomes; hand cards do not count.
* new_token: -2 when investing claims an unowned token or takes another player's
  token. Keeping an owned token or tying its owner gets 0. NO matching bonus.
* invested_dominance: -2 when investing if own INVESTMENTS BEFORE the action
  already reach (company_id + 1) // 2. Hand cards do not count here.
* offer_risk: after offering, compare own remaining hand+investments with the
  largest investment+known-hand count among opponents allowed to take that
  company (exclude its token owner). If not strictly ahead, -1 per remaining
  own card of that company; otherwise 0. Do not assume a future opponent pickup.

All terms add; penalties are preferences, never bans. Deck draw scores average
each component over unaccounted-for company copies, including the real market
fee and commitment bonus. This is an approximation, not knowledge of the deck:
the pool also contains removed cards and opponents' unknown hand cards.

RankingWeights names every tunable value. See docs/weighted_ranking_player.md for
worked examples, inspection, and the consequences of changing these weights.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import ClassVar, get_args

from simulator import Action, CompanyId, Observation, PlayerId

from ..base import ActionMask
from .ranking_base import RankingPlayer, ScoredAction, ScoreContributions


Weight = int | Fraction
COMPANIES = get_args(CompanyId)
POSITION_COMPONENTS = ("hand_cards", "hand_groups", "coins", "excess_holdings", "majority_position")
EVENT_COMPONENTS = ("pickup_commitment", "new_token", "invested_dominance", "offer_risk")


@dataclass(frozen=True)
class RankingWeights:
    """Immutable weights for all configurations using the shared scoring features.

    hand_values is ordered by company 5..10; group_values by hand count 0..4;
    majority_values by gaps <=-2, -1, 0, +1, >=+2. Other values multiply their
    event/count directly. Use signed values: default penalties are negative.
    Accept integers/Fractions only, excluding bool/floats, to keep ties exact.
    Use dataclasses.replace to create a modified configuration; reset preserves it.
    """

    hand_values: tuple[Weight, ...] = (1, 1, 0, 0, -1, -1)
    group_values: tuple[Weight, ...] = (0, 0, 1, 3, 5)
    majority_values: tuple[Weight, ...] = (-4, -2, 0, 1, 3)
    coin_value: Weight = Fraction(1, 2)
    excess_card_value: Weight = -1
    commitment_per_investment: Weight = 1
    new_token_value: Weight = -2
    dominated_investment_value: Weight = -2
    offer_risk_per_card: Weight = -1

    def __post_init__(self) -> None:
        """Reject mutable/mis-sized tables and inexact or nonnumeric weights."""
        sizes = {"hand_values": 6, "group_values": 5, "majority_values": 5}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in sizes:
                if not isinstance(value, tuple) or len(value) != sizes[field.name]:
                    raise ValueError(f"{field.name} must be a tuple of length {sizes[field.name]}")
                values = value
            else:
                values = (value,)
            if any(type(item) is not int and not isinstance(item, Fraction) for item in values):
                raise ValueError(f"{field.name} weights must be integers or Fractions")


class WeightedRankingPlayer(RankingPlayer):
    """Select by resulting-position value plus events, with seeded random ties.

    The only retained strategy data is an immutable weights object. No simulator,
    hidden-card access, history, or cached ranking is needed. Pass a dictionary
    of weight overrides to create a variant without subclassing. Omitted values
    keep their defaults. The mapping is copied into a validated, frozen
    RankingWeights object at construction, never consulted during decisions.
    New scoring features still require changes to the shared numeric methods.
    Normal gameplay calls _action_value; it does not compute a position baseline
    or create explanation records. rank_actions/explain_actions are opt-in.
    """

    DEFAULT_WEIGHTS: ClassVar[RankingWeights] = RankingWeights()

    def __init__(
        self, player_id: PlayerId, name: str = "Weighted ranking player", *,
        seed: int | None = None,
        weights: RankingWeights | Mapping[str, object] | None = None,
    ) -> None:
        """Initialize identity/RNG and normalize optional weight overrides once.

        Accept RankingWeights, a mapping of its field names to integer/Fraction
        weights (or immutable tuples for tables), or None for defaults. Unknown
        keys, invalid values, and unsupported containers raise ValueError.
        Copy the mapping; later caller edits cannot change this player's policy.
        """
        if isinstance(weights, Mapping):
            try:
                weights = RankingWeights(**weights)
            except TypeError as error:
                raise ValueError(f"Invalid ranking weight keys: {error}") from error
        elif weights is not None and not isinstance(weights, RankingWeights):
            raise ValueError("weights must be RankingWeights, a mapping, or None")
        super().__init__(player_id, name, seed=seed)
        self.weights = self.DEFAULT_WEIGHTS if weights is None else weights

    def position_components(self, observation: Observation) -> ScoreContributions:
        """Return the five raw current-position values, not action-event bonuses.

        Validate ownership; inactive/historical views are allowed. This is a
        heuristic position value, not the simulator's final score or an RL reward.
        Return immutable exact values without advancing RNG or storing the view.
        Used by explicit inspection/explanations, never by normal selection.
        """
        self._validate_observation(observation)
        own = next(state for state in observation.players if state.player_id == self.player_id)
        return tuple(zip(POSITION_COMPONENTS, self._position_values(
            observation, observation.hand, own.investments, own.coins,
        )))

    def explain_actions(self, observation: Observation, mask: ActionMask) -> tuple[ScoredAction, ...]:
        """Opt in to gain/loss breakdowns relative to the current position.

        Return every legal action in the same order as rank_actions. Evaluate
        the current-position baseline ONCE for the whole request, then subtract
        its five components from each action's resulting-position components.
        Event adjustments stay unchanged. Each returned score is a gain, equal
        to selection score minus the shared baseline; this preserves all ties.
        For deck draws, subtract after averaging the possible outcomes.

        Neither the game loop nor policy calls this method. Use it for debugging
        only when explanations are wanted. No randomness or retained state is
        involved, so inspection cannot change subsequent seeded decisions.
        """
        ranked = self.rank_actions(observation, mask)
        baseline = dict(self.position_components(observation))
        return tuple(ScoredAction(entry.action, tuple(
            (name, value - baseline.get(name, 0)) for name, value in entry.contributions
        )) for entry in ranked)

    def estimate_draw_probabilities(self, observation: Observation) -> tuple[tuple[CompanyId, Fraction], ...]:
        """Estimate a draw by normalizing all six unaccounted-for copy counts.

        Remaining_i = i - own_hand_i - all_investments_i - market_i
                        - other_players_known_hand_i.
        Do not subtract own known counts again. Zero-count companies receive
        probability zero. The denominator is the whole unknown pool, NOT deck
        size. Removed cards and unknown opponent cards remain in that pool.
        Never inspect actual deck/removed cards or complete opposing hands.
        Require a nonempty deck and consistent nonnegative remaining counts.
        """
        self._validate_observation(observation)
        if observation.deck_size <= 0:
            raise ValueError("A draw estimate requires a nonempty deck")
        counts = {company: company - observation.hand.count(company) for company in COMPANIES}
        for state in observation.players:
            known = dict(state.known_hand_counts)
            for company in COMPANIES:
                counts[company] -= state.investments.count(company)
                if state.player_id != self.player_id:
                    counts[company] -= known[company]
        for card in observation.market:
            counts[card.company_id] -= 1
        total = sum(counts.values())
        if total <= 0 or any(count < 0 for count in counts.values()):
            raise ValueError("Observation has inconsistent unaccounted-for card counts")
        return tuple((company, Fraction(counts[company], total)) for company in COMPANIES)

    def _position_values(
        self, observation: Observation, hand: tuple[CompanyId, ...],
        investments: tuple[CompanyId, ...], coins: int,
    ) -> tuple[Fraction, ...]:
        """Evaluate supplied own holdings/balance against unchanged public opponents.

        Hypothetical own values are local data, not a fabricated full simulator.
        Hand groups use exact hand counts; majority/excess use combined counts.
        Majority compares all opponents, including the token owner, and skips
        companies with zero own holdings. Opponent known hands are lower bounds.
        Return numeric values in POSITION_COMPONENTS order, without labels.
        """
        weights = self.weights
        hand_value = groups = excess = majority = Fraction(0)
        for index, company in enumerate(COMPANIES):
            hand_count = hand.count(company)
            own_count = hand_count + investments.count(company)
            hand_value += hand_count * weights.hand_values[index]
            groups += weights.group_values[hand_count]
            excess += max(0, own_count - (company + 1) // 2) * weights.excess_card_value
            if own_count:
                rival_count = self._largest_opponent_holding(observation, company)
                gap = max(-2, min(2, own_count - rival_count))
                majority += weights.majority_values[gap + 2]
        return (
            hand_value, groups, coins * Fraction(weights.coin_value), excess, majority,
        )

    def _largest_opponent_holding(
        self, observation: Observation, company: CompanyId, *, eligible_only: bool = False,
    ) -> int:
        """Return the maximum public investment+known-hand count, never their sum.

        For offer risk only, exclude the opposing token owner, who cannot take
        this company. Do not add the offered card to any opponent's holding.
        """
        owner = dict(observation.anti_monopoly_owners)[company]
        return max((
            state.investments.count(company) + dict(state.known_hand_counts)[company]
            for state in observation.players
            if state.player_id != self.player_id
            and (not eligible_only or state.player_id != owner)
        ), default=0)

    def _score_action(self, observation: Observation, action: Action) -> ScoreContributions:
        """Name the numeric selection components for explicit rank_actions inspection.

        All nine names remain present, including zeros. Position components are
        resulting values, NOT changes from the current position. Events are
        one-off adjustments. Normal selection uses _action_value directly.
        """
        return tuple(zip(POSITION_COMPONENTS + EVENT_COMPONENTS, self._action_values(observation, action)))

    def _action_value(self, observation: Observation, action: Action) -> Fraction:
        """Sum numeric selection features without baselines or explanation objects."""
        return sum(self._action_values(observation, action), Fraction(0))

    def _action_values(self, observation: Observation, action: Action) -> tuple[Fraction, ...]:
        """Share numeric feature arithmetic between selection and inspection.

        Return resulting-position values followed by event values. Evaluate
        each hypothetical draw before averaging its nine values: nonlinear
        group/excess/majority features cannot be applied to an average hand.
        Never evaluate the current-position baseline here.
        """
        if action.kind != "draw_deck":
            return self._outcome_values(observation, action)
        expected = [Fraction(0)] * (len(POSITION_COMPONENTS) + len(EVENT_COMPONENTS))
        for company, probability in self.estimate_draw_probabilities(observation):
            if probability:
                for index, value in enumerate(self._outcome_values(observation, action, drawn_company=company)):
                    expected[index] += probability * value
        return tuple(expected)

    def _outcome_values(
        self, observation: Observation, action: Action, *, drawn_company: CompanyId | None = None,
    ) -> tuple[Fraction, ...]:
        """Project one immediate outcome locally and add the four event adjustments.

        Deck fees count individual nonexempt market cards, even at negative
        balances. Market pickups collect all attached coins. Investment events
        use BEFORE counts; offer risk uses AFTER own counts. Do not execute a
        simulator step, reveal terminal hands, or plan the next action.
        Read current counts for event conditions, but do not evaluate their
        aggregate position value. Return numeric features without labels.
        """
        own = next(state for state in observation.players if state.player_id == self.player_id)
        hand, investments, coins = list(observation.hand), list(own.investments), own.coins
        commitment = new_token = invested_dominance = offer_risk = Fraction(0)
        weights = self.weights
        if action.kind in ("take_market", "draw_deck"):
            if action.kind == "take_market":
                card = observation.market[action.market_index]
                company = card.company_id
                coins += card.coins
            else:
                if drawn_company is None:
                    raise ValueError("A projected deck draw requires a company")
                company = drawn_company
                owners = dict(observation.anti_monopoly_owners)
                coins -= sum(owners[card.company_id] != self.player_id for card in observation.market)
            hand.append(company)
            commitment = Fraction(investments.count(company) * weights.commitment_per_investment)
        else:
            company = action.company_id
            hand.remove(company)
            if action.kind == "invest":
                if self.would_acquire_anti_monopoly(observation, company):
                    new_token = Fraction(weights.new_token_value)
                if investments.count(company) >= (company + 1) // 2:
                    invested_dominance = Fraction(weights.dominated_investment_value)
                investments.append(company)
            elif action.kind == "offer_to_market":
                remaining = hand.count(company) + investments.count(company)
                rival = self._largest_opponent_holding(observation, company, eligible_only=True)
                if remaining <= rival:
                    offer_risk = Fraction(remaining * weights.offer_risk_per_card)
            else:
                raise ValueError(f"Unsupported action kind: {action.kind}")
        after = self._position_values(observation, tuple(hand), tuple(investments), coins)
        return after + (commitment, new_token, invested_dominance, offer_risk)
