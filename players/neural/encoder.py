"""Version 1 of the 332-value, viewer-relative Startups observation encoding.

Only public quantities and the viewer's exact hand are read. Card-list ordering
within hands/investments is intentionally discarded; market ordering is kept.
No action mask, hidden state, policy scores, or retained history is included.
See docs/neural_observation_encoder.md for the exact schema and completeness audit.
"""

from array import array
from collections import Counter
from math import isfinite
from types import MappingProxyType

from simulator import CompanyId, MarketCard, Observation, PlayerId, PublicPlayerState


class FlatObservationEncoder:
    """Stateless callable returning a fresh array('f') of 332 float32 values.

    encode(observation) and encoder(observation) are equivalent. Accept any
    player's own Observation, including historical, inactive, and terminal views.
    Keep the viewer fixed when encoding a history sequence. Relative seat 0 is
    that viewer; seats 1..3 follow the cyclic game order. The global phase never
    changes merely because the viewer is inactive.

    Validate representable field shapes/types, without reimplementing the rules
    or checking full card conservation/reachability. Never truncate, clip, mutate
    the observation, or query a Simulator. Output buffers belong to the caller.
    The standard library's C-float array is required to use four-byte items.
    """

    NAME = "startups.flat_observation"
    VERSION = 1
    SIZE = 332
    DTYPE = "float32"
    COMPANY_IDS: tuple[CompanyId, ...] = (5, 6, 7, 8, 9, 10)
    PLAYER_IDS: tuple[PlayerId, ...] = (1, 2, 3, 4)
    PHASES = ("take", "play", "finished")
    MAX_MARKET_CARDS = 28
    PLAYER_WIDTH = 14
    TOKEN_WIDTH = 5
    MARKET_WIDTH = 8
    COIN_SCALE = 15
    HAND_SIZE_SCALE = 4
    DECK_SIZE_SCALE = 28

    # Python slice stops are exclusive. These labels are checkpoint schema names.
    SLICES = MappingProxyType({
        "phase": slice(0, 3),
        "deck_size": slice(3, 4),
        "taken_market_company": slice(4, 11),
        "active_player": slice(11, 16),
        "hand_counts": slice(16, 22),
        "players": slice(22, 78),
        "anti_monopoly_owners": slice(78, 108),
        "market": slice(108, 332),
    })

    def __call__(self, observation: Observation) -> array:
        """Adapt encode to the callable ObservationEncoder interface."""
        return self.encode(observation)

    @classmethod
    def metadata(cls) -> dict[str, str | int]:
        """Return fresh JSON-compatible schema metadata for a future checkpoint.

        A loader should compare this entire record with the checkpoint's stored
        observation_encoding record. This does not load or validate checkpoints;
        the controller and training/checkpoint machinery remain separate.
        """
        return {"name": cls.NAME, "version": cls.VERSION, "size": cls.SIZE, "dtype": cls.DTYPE}

    @classmethod
    def encode(cls, observation: Observation) -> array:
        """Encode one visible snapshot using the frozen version-1 layout.

        Card counts are divided by company ID; balances and market coins by 15;
        hand sizes by 4; deck size by 28. Coin features may be negative or exceed
        one. Owner/active-player categories are [none, self, next, following,
        last]; taken_market_company is [none, 5, 6, 7, 8, 9, 10]. This encoded
        field reads Observation.taken_company_this_turn without renaming the
        simulator's existing field. Phase order is [take, play, finished].

        A player block is [six investment counts, six public known-hand counts,
        balance, hand size]. Keep the viewer's public counts separate from their
        exact hand. A market block is [present, six company indicators, coins].
        Unused market blocks are zero; occupied but illegal pickups stay present.
        Raise ValueError on malformed/unrepresentable fields, or TypeError if
        the input is not an Observation. No active-turn requirement is imposed.
        """
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        viewer = cls._player_id(observation.player_id, "player_id")
        active = cls._player_id(observation.current_player_id, "current_player_id", optional=True)
        if type(observation.phase) is not str or observation.phase not in cls.PHASES:
            raise ValueError("phase must be take, play, or finished")
        cls._integer(observation.deck_size, "deck_size", minimum=0, maximum=28)
        taken = observation.taken_company_this_turn
        if taken is not None:
            cls._company_id(taken)
        hand = cls._card_counts(observation.hand, "hand", max_size=4)

        values = array("f", [0.0]) * cls.SIZE
        if values.itemsize != 4:
            raise RuntimeError("This encoder requires four-byte array('f') elements")
        values[cls.SLICES["phase"].start + cls.PHASES.index(observation.phase)] = 1
        values[cls.SLICES["deck_size"].start] = observation.deck_size / cls.DECK_SIZE_SCALE
        taken_offset = 0 if taken is None else taken - 4
        values[cls.SLICES["taken_market_company"].start + taken_offset] = 1
        values[cls.SLICES["active_player"].start + cls._owner_offset(active, viewer)] = 1
        for index, company in enumerate(cls.COMPANY_IDS):
            values[cls.SLICES["hand_counts"].start + index] = hand[company] / company

        if not isinstance(observation.players, tuple) or len(observation.players) != 4:
            raise ValueError("players must contain exactly four public player records")
        seen_players = set()
        for player in observation.players:
            if not isinstance(player, PublicPlayerState):
                raise ValueError("players must contain PublicPlayerState records")
            pid = cls._player_id(player.player_id, "public player_id")
            if pid in seen_players:
                raise ValueError("players must contain each player_id exactly once")
            seen_players.add(pid)
            cls._integer(player.hand_size, "hand_size", minimum=0, maximum=4)
            investments = cls._card_counts(player.investments, "investments")
            known = cls._company_pairs(player.known_hand_counts, "known_hand_counts")
            for count in known.values():
                cls._integer(count, "known_hand_counts count", minimum=0, maximum=player.hand_size)
            if sum(known.values()) > player.hand_size:
                raise ValueError("known_hand_counts cannot exceed hand_size")
            if pid == viewer and (
                player.hand_size != len(observation.hand)
                or any(known[company] > hand[company] for company in cls.COMPANY_IDS)
            ):
                raise ValueError("viewer's public hand information contradicts their exact hand")
            base = cls.SLICES["players"].start + cls.PLAYER_WIDTH * ((pid - viewer) % 4)
            for index, company in enumerate(cls.COMPANY_IDS):
                values[base + index] = investments[company] / company
                values[base + 6 + index] = known[company] / company
            values[base + 12] = cls._coins(player.coins, "coins")
            values[base + 13] = player.hand_size / cls.HAND_SIZE_SCALE

        owners = cls._company_pairs(observation.anti_monopoly_owners, "anti_monopoly_owners")
        for index, company in enumerate(cls.COMPANY_IDS):
            owner = cls._player_id(owners[company], "token owner", optional=True)
            base = cls.SLICES["anti_monopoly_owners"].start + cls.TOKEN_WIDTH * index
            values[base + cls._owner_offset(owner, viewer)] = 1

        if not isinstance(observation.market, tuple) or len(observation.market) > cls.MAX_MARKET_CARDS:
            raise ValueError("market must be a tuple of at most 28 cards")
        for index, card in enumerate(observation.market):
            if not isinstance(card, MarketCard):
                raise ValueError("market must contain MarketCard records")
            company = cls._company_id(card.company_id)
            base = cls.SLICES["market"].start + cls.MARKET_WIDTH * index
            values[base] = 1
            values[base + 1 + company - 5] = 1
            values[base + 7] = cls._coins(card.coins, "market coins", minimum=0)
        return values

    @classmethod
    def _player_id(cls, value: object, label: str, *, optional: bool = False) -> PlayerId | None:
        """Validate an absolute player ID before converting it to a relative seat."""
        if optional and value is None:
            return None
        if type(value) is not int or value not in cls.PLAYER_IDS:
            raise ValueError(f"{label} must be a player ID from 1 through 4")
        return value

    @classmethod
    def _company_id(cls, value: object) -> CompanyId:
        """Reject invalid company indices, including bool/float integer lookalikes."""
        if type(value) is not int or value not in cls.COMPANY_IDS:
            raise ValueError("company ID must be an integer from 5 through 10")
        return value

    @staticmethod
    def _owner_offset(owner: PlayerId | None, viewer: PlayerId) -> int:
        """Map a validated owner/active seat to [none, self, next, following, last]."""
        return 0 if owner is None else 1 + (owner - viewer) % 4

    @staticmethod
    def _integer(value: object, label: str, *, minimum: int | None = None, maximum: int | None = None) -> None:
        """Validate integer quantities without accepting booleans or silently clipping."""
        if type(value) is not int or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            raise ValueError(f"{label} must be an integer in the supported range")

    @classmethod
    def _card_counts(cls, cards: object, label: str, *, max_size: int | None = None) -> Counter:
        """Count an immutable card sequence, checking company IDs and supply bounds."""
        if not isinstance(cards, tuple) or (max_size is not None and len(cards) > max_size):
            raise ValueError(f"{label} must be a card tuple of a supported size")
        counts = Counter(cls._company_id(card) for card in cards)
        if any(count > company for company, count in counts.items()):
            raise ValueError(f"{label} contains more copies than a company's supply")
        return counts

    @classmethod
    def _company_pairs(cls, pairs: object, label: str) -> dict:
        """Index exactly six (company, value) pairs, independently of pair order."""
        if not isinstance(pairs, tuple) or len(pairs) != 6:
            raise ValueError(f"{label} must contain six company pairs")
        result = {}
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"{label} must contain (company, value) pairs")
            company = cls._company_id(pair[0])
            if company in result:
                raise ValueError(f"{label} contains a duplicate company")
            result[company] = pair[1]
        return result

    @classmethod
    def _coins(cls, value: object, label: str, *, minimum: int | None = None) -> float:
        """Scale integer coins, preserving signs and rejecting float32 overflow."""
        cls._integer(value, label, minimum=minimum)
        try:
            scaled = value / cls.COIN_SCALE
            if not isfinite(scaled) or abs(scaled) > 3.4028234663852886e38:
                raise ValueError(f"{label} cannot be represented as finite float32")
        except OverflowError as error:
            raise ValueError(f"{label} cannot be represented as finite float32") from error
        return scaled
