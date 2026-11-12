"""Shared operations and decision hooks for structured rule-based policies.

StructuredRuleBasedPlayer inherits counting and public token-acquisition helpers
from RuleBasedPlayer and adds legal-action filtering and phase dispatch. Concrete
policies supply four abstract decision hooks. The hooks return choices only;
Simulator.step remains responsible for all game mutations.
RuleBasedPlayer stays available for policies such as the uniform random baseline
whose decisions do not need this decomposition.

These basic operations are implemented and shared by every policy subclass.
The four abstract decision hooks must be implemented by each strategy.
"""

from abc import abstractmethod
from typing import Literal, get_args

from action_codec import ActionCodec
from simulator import Action, ActionKind, CompanyId, Observation

from ..base import ActionMask
from .base import RuleBasedPlayer


PlayDecision = Literal["invest", "offer_to_market"]


class StructuredRuleBasedPlayer(RuleBasedPlayer):
    """Define a template for policies that choose a source, then a play type/card.

    Inherit identity, kind, reset, observation helpers/callbacks, and the async
    choose_action wrapper from RuleBasedPlayer. Subclasses implement
    take_from_market, decide_invest_or_market, invest, and put_to_market.
    They ordinarily inherit policy and the basic operations below.

    Each hook receives the current observation and matching legal-action mask.
    Thresholds, priorities, and tie-breaking belong to the subclass. Hooks must
    return legal choices, must not advance the game, and must not consume hidden
    simulator state. Any retained context must be cleared by a reset override.
    """

    def legal_actions_of_kind(self, mask: ActionMask, kind: ActionKind) -> tuple[Action, ...]:
        """Decode mask-enabled actions of one kind in ascending ActionCodec ID order.

        Validate the decision mask using _validate_mask and reject unknown kinds
        with ValueError. An all-False mask is invalid: there is no decision to
        make. Use ActionCodec rather than duplicating action encoding.
        Return an empty tuple if a valid mask has no actions of this kind.
        Preserve each distinct market slot and one action per company for
        investments/offers.
        The simulator's mask determines legality; this helper adds no strategy
        preferences or game rules and has no side effects.
        """
        self._validate_mask(mask)
        if not isinstance(kind, str) or kind not in get_args(ActionKind):
            raise ValueError(f"Unknown action kind: {kind!r}")
        actions: list[Action] = []
        for action_id, allowed in enumerate(mask):
            if allowed:
                action = ActionCodec.decode(action_id)
                if action.kind == kind:
                    actions.append(action)
        return tuple(actions)

    def policy(self, observation: Observation, mask: ActionMask) -> int:
        """Coordinate the appropriate hooks and return a fixed ActionCodec ID.

        Validate the current request before invoking hooks. In take phase, call
        take_from_market: None means draw_deck, while an integer means take_market
        at that zero-based index. Index 0 is a valid market choice, not a deck draw.

        In play phase, call decide_invest_or_market, then exactly one of invest
        or put_to_market to obtain the chosen company ID. Construct the matching
        Action and encode it using ActionCodec. Reject unknown play decisions,
        malformed hook results (including boolean indices/companies), and masked
        choices with ValueError. Never substitute another action silently.

        The inherited choose_action also validates the request and final result.
        This dispatcher owns phase control and encoding; subclasses own strategy
        preferences through their hooks. Do not call on_observation or step.
        Hook exceptions propagate without retries or fallback choices.
        """
        self._validate_request(observation, mask)
        if observation.phase == "take":
            market_index = self.take_from_market(observation, mask)
            action = (
                Action("draw_deck") if market_index is None
                else Action("take_market", market_index=market_index)
            )
        else:
            decision = self.decide_invest_or_market(observation, mask)
            if type(decision) is not str or decision not in get_args(PlayDecision):
                raise ValueError("decide_invest_or_market must return 'invest' or 'offer_to_market'")
            if decision == "invest":
                action = Action("invest", company_id=self.invest(observation, mask))
            else:
                action = Action("offer_to_market", company_id=self.put_to_market(observation, mask))

        return self._validate_action_id(ActionCodec.encode(action), mask)

    @abstractmethod
    def take_from_market(self, observation: Observation, mask: ActionMask) -> int | None:
        """Choose a legal market index, or None to draw from the deck.

        Called only in take phase. Return a zero-based index into the current
        observation.market, not a permanent card identifier, company ID, menu
        number, or ActionCodec ID. Return None only when draw_deck is legal.
        Select only a mask-enabled pickup; owning its company's anti-monopoly
        token makes that pickup illegal. Return the choice without taking a card.
        """
        raise NotImplementedError

    @abstractmethod
    def decide_invest_or_market(self, observation: Observation, mask: ActionMask) -> PlayDecision:
        """Choose 'invest' or 'offer_to_market' during play phase.

        Choose a type for which at least one legal action is available. The
        corresponding card-selection hook will be called next with the same
        observation and mask. Shared strategy-specific candidate calculations
        can live in private subclass helpers used by both hooks. Do not remove,
        invest, or offer a card here.
        """
        raise NotImplementedError

    @abstractmethod
    def invest(self, observation: Observation, mask: ActionMask) -> CompanyId:
        """Choose the company of one card to invest, without executing the move.

        Called only after decide_invest_or_market returns 'invest'. Return a
        company ID from 5 through 10 whose investment action is mask-enabled.
        Apply the concrete policy's eligibility and tie-breaking preferences.
        The returned value is a company ID, not an ActionCodec ID or hand index.
        """
        raise NotImplementedError

    @abstractmethod
    def put_to_market(self, observation: Observation, mask: ActionMask) -> CompanyId:
        """Choose the company of one card to offer, without executing the move.

        Called only after decide_invest_or_market returns 'offer_to_market'.
        Return a company ID from 5 through 10 whose offer action is mask-enabled,
        respecting the restriction on the company taken from the market this
        turn. Apply the policy's offer preferences; do not alter cards or coins.
        """
        raise NotImplementedError
