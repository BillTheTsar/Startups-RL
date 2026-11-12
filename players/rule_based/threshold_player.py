"""One deterministic threshold family, with an optional chicken investment filter.

During take, first consider legal market cards whose company has at least two
copies in hand OR at least three copies across hand and investments, counted
before the pickup. Both conditions belong to the same priority group. If that
group is empty, consider any legal market card with at least three coins. Rank
either group by lowest company ID, then most coins, then lowest market index.
Otherwise draw from the deck, regardless of balance or market fees.

During play, invest if any legal company has at least company_id // 3 copies in
the current hand, including the newly taken card and excluding investments.
Prefer the lowest qualifying company ID. Otherwise offer a legal card from the
company with the fewest copies across hand and investments, breaking ties by
highest company ID. Count holdings before offering. Use these rules on the final
turn too. The policy uses only the current observation and matching action mask.
The numbers above are profile 1's defaults; ThresholdParameters supplies the
other profiles without subclassing. With chicken=True, reject new-token
investments BEFORE the hand threshold, unless no legal offer is available.
Keeping an existing token or tying its owner is safe. All other rules stay the
same, including the hand threshold on a forced investment.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields

from action_codec import ActionCodec
from simulator import CompanyId, Observation, PlayerId

from ..base import ActionMask
from .structured_base import PlayDecision, StructuredRuleBasedPlayer


@dataclass(frozen=True)
class ThresholdParameters:
    """Immutable numeric settings shared by ordinary and chicken profiles.

    Pickup minima are nonnegative integers. The investment divisor is a positive
    integer. Booleans/floats are rejected. Hand and combined pickup counts are
    measured before taking; investment eligibility uses the resulting hand only.
    """

    hand_pickup_threshold: int = 2
    own_pickup_threshold: int = 3
    coin_pickup_threshold: int = 3
    investment_divisor: int = 3

    def __post_init__(self) -> None:
        """Validate settings once at construction, never during policy evaluation."""
        for field in fields(self):
            value = getattr(self, field.name)
            minimum = 1 if field.name == "investment_divisor" else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{field.name} must be an integer >= {minimum}")


class ThresholdPlayer(StructuredRuleBasedPlayer):
    """Implement four decision hooks using shared counting, filtering, and dispatch.

    No RNG, simulator reference, or observation history is retained. The inherited
    reset preserves identity/settings and ignores its seed. All choices respect
    the mask. The ordinary profile ignores opponents' holdings; chicken mode
    uses only public investments and token ownership to predict acquisition.
    No policy reads opponents' hidden hands or retains observation history.
    """

    def __init__(
        self, player_id: PlayerId, name: str = "Threshold player", *,
        parameters: ThresholdParameters | Mapping[str, object] | None = None,
        chicken: bool = False,
    ) -> None:
        """Copy optional dictionary overrides into frozen settings and set chicken mode.

        Omitted values use profile 1's defaults. Accept a ThresholdParameters
        object directly, or None for defaults. Reject invalid keys/settings and
        nonboolean chicken flags. Later dictionary edits cannot affect this player.
        """
        if type(chicken) is not bool:
            raise ValueError("chicken must be a boolean")
        if isinstance(parameters, Mapping):
            try:
                parameters = ThresholdParameters(**parameters)
            except TypeError as error:
                raise ValueError(f"Invalid threshold parameter keys: {error}") from error
        elif parameters is not None and not isinstance(parameters, ThresholdParameters):
            raise ValueError("parameters must be ThresholdParameters, a mapping, or None")
        super().__init__(player_id, name)
        self.parameters = ThresholdParameters() if parameters is None else parameters
        self.chicken = chicken

    def take_from_market(self, observation: Observation, mask: ActionMask) -> int | None:
        """Choose a qualifying collection pickup, then a coin pickup, then a draw.

        Collection requires hand count >= parameters.hand_pickup_threshold OR
        combined count >= parameters.own_pickup_threshold, before pickup.
        Coin fallback requires >= parameters.coin_pickup_threshold attached coins and
        applies to duplicate and distinct hands. Version 1 uses 2, 3, and 3,
        respectively. Only mask-enabled market slots enter either group. Return
        the selected zero-based index, or None for a legal deck draw. Raise ValueError if neither
        a qualifying pickup nor a deck draw is available; never relax the rules.
        """
        hand_counts = self.count_hand_cards(observation)
        own_counts = self.count_own_cards(observation)
        market_indices = tuple(
            action.market_index for action in self.legal_actions_of_kind(mask, "take_market")
            if action.market_index is not None
        )
        collection = tuple(
            index for index in market_indices
            if hand_counts[observation.market[index].company_id] >= self.parameters.hand_pickup_threshold
            or own_counts[observation.market[index].company_id] >= self.parameters.own_pickup_threshold
        )
        if collection:
            return self._select_market_index(observation, collection)

        coin_pickups = tuple(
            index for index in market_indices
            if observation.market[index].coins >= self.parameters.coin_pickup_threshold
        )
        if coin_pickups:
            return self._select_market_index(observation, coin_pickups)
        if self.legal_actions_of_kind(mask, "draw_deck"):
            return None
        raise ValueError("No qualifying market pickup or legal deck draw is available")

    def decide_invest_or_market(self, observation: Observation, mask: ActionMask) -> PlayDecision:
        """Invest if a company passes the configured filters; otherwise offer.

        Chicken mode filters new-token acquisitions before testing the hand
        threshold, with a fallback only if no offer is legal. Existing
        investments never contribute to the hand threshold. Raise ValueError
        if no company qualifies and no offer is legal, without bypassing the
        strategy's threshold. The corresponding hook chooses the company next.
        """
        if self._investment_candidates(observation, mask):
            return "invest"
        if self.legal_actions_of_kind(mask, "offer_to_market"):
            return "offer_to_market"
        raise ValueError("No qualifying investment or legal market offer is available")

    def invest(self, observation: Observation, mask: ActionMask) -> CompanyId:
        """Return the lowest qualifying company, or raise ValueError if none exists."""
        candidates = self._investment_candidates(observation, mask)
        if not candidates:
            raise ValueError("No legal company meets the investment threshold")
        return candidates[0]

    def put_to_market(self, observation: Observation, mask: ActionMask) -> CompanyId:
        """Offer from the smallest total own holding, breaking ties by highest ID.

        Count this player's hand plus investments before offering. The mask
        excludes the company just taken from the market. Raise ValueError when
        no offer is legal. Return a company ID without removing any card.
        """
        own_counts = self.count_own_cards(observation)
        companies = tuple(
            action.company_id for action in self.legal_actions_of_kind(mask, "offer_to_market")
            if action.company_id is not None
        )
        if not companies:
            raise ValueError("No legal market offer is available")
        return min(companies, key=lambda company: (own_counts[company], -company))

    def _investment_candidates(
        self, observation: Observation, mask: ActionMask
    ) -> tuple[CompanyId, ...]:
        """Apply optional chicken risk filtering, then the configured hand threshold.

        In chicken mode, remove investments that acquire a new token before
        threshold selection. Keeping a token or tying its owner is safe. If no
        safe company qualifies, offer when possible. Only if no offer is legal,
        fall back to the ordinary threshold-qualified investments. Do not relax
        the hand threshold for restricted masks. Preserve the supplied mask.
        """
        if not self.chicken:
            return self._qualifying_investments(observation, mask)
        self._validate_observation(observation)
        safe_mask = list(mask)
        for action in self.legal_actions_of_kind(mask, "invest"):
            if self.would_acquire_anti_monopoly(observation, action.company_id):
                safe_mask[ActionCodec.encode(action)] = False
        candidates = (
            self._qualifying_investments(observation, tuple(safe_mask))
            if any(safe_mask) else ()
        )
        if candidates or self.legal_actions_of_kind(mask, "offer_to_market"):
            return candidates
        return self._qualifying_investments(observation, mask)

    def _qualifying_investments(
        self, observation: Observation, mask: ActionMask
    ) -> tuple[CompanyId, ...]:
        """Return legal companies meeting hand count >= company_id // investment_divisor.

        Version 1 uses divisor 3: 1 card for company 5, 2 for companies 6 through
        8, and 3 for companies 9 and 10. Variants may override the divisor.
        The shared action filter supplies ascending company order and no
        duplicates. Return () if none qualifies. Recompute from the
        current hand for both investment hooks, without caching a pending choice.
        """
        hand_counts = self.count_hand_cards(observation)
        return tuple(
            action.company_id for action in self.legal_actions_of_kind(mask, "invest")
            if action.company_id is not None
            and hand_counts[action.company_id] >= action.company_id // self.parameters.investment_divisor
        )

    def _select_market_index(
        self, observation: Observation, market_indices: tuple[int, ...]
    ) -> int | None:
        """Rank a legal candidate group by company, descending coins, then index.

        Candidate indices already meet the chosen priority group's conditions.
        Return None for an empty group. Selection does not depend on the order
        of the input indices, and a higher company's coins never override the
        preference for a lower company ID.
        """
        return min(
            market_indices,
            key=lambda index: (
                observation.market[index].company_id,
                -observation.market[index].coins,
                index,
            ),
            default=None,
        )
