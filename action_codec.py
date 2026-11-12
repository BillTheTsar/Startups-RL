"""Translate game actions to a stable model interface for training and inference.

Use the same ActionCodec version with a trained policy and its saved experiences.
Market slots refer to indices in that decision's Observation.market, so record
the observation, selected action ID, and mask together. Recompute the mask for
every decision, including the play decision immediately after taking a card.

Masks are immutable tuples of Python booleans (True means legal), ready to be
converted to arrays or tensors by the training framework. Exclude False entries
before sampling or taking an argmax. An all-False mask means no decision should
be requested, as with an inactive player or a finished game.
"""

from collections.abc import Iterable
from operator import index
from typing import SupportsIndex

from simulator import Action, CompanyId


class ActionCodec:
    """Stateless encoding of 41 action slots, independent of any model library.

    Version 1 assigns 0 to draw, 1..28 to market indices 0..27, 29..34 to
    investing in companies 5..10, and 35..40 to offering companies 5..10.
    The 28-slot market bound is 45 cards - 5 removed - 4 players * 3 hand cards.

    These IDs describe the full vocabulary, not the currently legal choices.
    Encoding/decoding never checks a game state. Generate a mask exclusively
    from Simulator.get_legal_actions; the codec contains no duplicate game rules.
    """

    VERSION = 1
    MAX_MARKET_CARDS = 28
    ACTION_COUNT = 41
    _COMPANY_IDS: tuple[CompanyId, ...] = (5, 6, 7, 8, 9, 10)
    _MARKET_START = 1
    _INVEST_START = 29
    _OFFER_START = 35

    @classmethod
    def encode(cls, action: Action) -> int:
        """Return the stable ID for a well-formed action within this vocabulary.

        Raise ValueError for unknown kinds, missing/extraneous arguments, or
        arguments outside the supported ranges. Booleans and floats are not
        company IDs or market indices. Encoding does not establish legality.
        """
        if not isinstance(action, Action):
            raise ValueError("action must be an Action instance")

        if action.kind == "draw_deck":
            if action.company_id is not None or action.market_index is not None:
                raise ValueError("draw_deck must not have a company_id or market_index")
            return 0

        if action.kind == "take_market":
            if (
                action.company_id is not None
                or type(action.market_index) is not int
                or not 0 <= action.market_index < cls.MAX_MARKET_CARDS
            ):
                raise ValueError("take_market requires only a market_index from 0 through 27")
            return cls._MARKET_START + action.market_index

        if action.kind in ("invest", "offer_to_market"):
            if (
                action.market_index is not None
                or type(action.company_id) is not int
                or action.company_id not in cls._COMPANY_IDS
            ):
                raise ValueError("invest and offer_to_market require only a company_id from 5 through 10")
            offset = cls._INVEST_START if action.kind == "invest" else cls._OFFER_START
            return offset + cls._COMPANY_IDS.index(action.company_id)

        raise ValueError(f"Unknown action kind: {action.kind!r}")

    @classmethod
    def decode(cls, action_id: SupportsIndex) -> Action:
        """Return the Action for an ID from 0 through 40, without checking legality.

        Accept Python integers and integer-like model outputs implementing
        __index__, without importing a numerical library. Reject booleans,
        noninteger values, and out-of-range IDs with ValueError.
        """
        if isinstance(action_id, bool):
            raise ValueError("action_id must be an integer from 0 through 40")
        try:
            action_id = index(action_id)
        except TypeError as error:
            raise ValueError("action_id must be an integer from 0 through 40") from error

        if not 0 <= action_id < cls.ACTION_COUNT:
            raise ValueError("action_id must be an integer from 0 through 40")
        if action_id == 0:
            return Action("draw_deck")
        if action_id < cls._INVEST_START:
            return Action("take_market", market_index=action_id - cls._MARKET_START)
        if action_id < cls._OFFER_START:
            return Action("invest", company_id=cls._COMPANY_IDS[action_id - cls._INVEST_START])
        return Action("offer_to_market", company_id=cls._COMPANY_IDS[action_id - cls._OFFER_START])

    @classmethod
    def make_mask(cls, legal_actions: Iterable[Action]) -> tuple[bool, ...]:
        """Encode supplied legal actions as an immutable mask of exactly 41 bools.

        Empty input yields all False. Duplicate actions are harmless and input
        order does not affect the mask. Malformed actions raise ValueError rather
        than being silently omitted. This method trusts the simulator's legality
        decisions and does not query or modify a game or an agent.
        """
        mask = [False] * cls.ACTION_COUNT
        for action in legal_actions:
            mask[cls.encode(action)] = True
        return tuple(mask)
