"""
Design principles:
- Do not give Players a reference to the simulator. A game loop should pass observations to players.
- The simulator owns hands and balances, keyed by player ID; Player objects only choose actions.

| Information | Simulator stores | Player observes |
|---|---|---|
| Draw deck | Complete ordered cards | Number remaining |
| Five removed cards | Their identities | Only that five were removed |
| Hands | Every player’s cards | Own cards and opponents’ hand sizes |
| Investments | All played cards | All played cards |
| Market | Each card and its attached coins | Same |
| Coin balances | All four balances | Same |
| Anti-Monopoly chips | Current owners | Same |
| Turn | Active player and action phase | Same |
| Current-turn restriction | Company taken from market, if any | Same |
| Random generator | Internal state | Nothing |
"""

from dataclasses import dataclass
from random import Random
from typing import Literal


PlayerId = Literal[1, 2, 3, 4]
CompanyId = Literal[5, 6, 7, 8, 9, 10]
Phase = Literal["take", "play", "finished"]
ActionKind = Literal["draw_deck", "take_market", "invest", "offer_to_market"]


@dataclass(frozen=True)
class Action:
    """A proposed move, containing only its kind and the relevant argument.

    ``draw_deck`` uses neither argument; ``take_market`` uses ``market_index``;
    ``invest`` and ``offer_to_market`` use ``company_id``. The simulator must
    reject missing, extraneous, or invalid arguments when validating a move.
    """

    kind: ActionKind
    company_id: CompanyId | None = None
    market_index: int | None = None


@dataclass(frozen=True)
class MarketCard:
    """One market entry and its attached coins, distinct from other entries.

    Entries are immutable: the simulator replaces an entry when coins change.
    """

    company_id: CompanyId
    coins: int


@dataclass(frozen=True)
class PublicPlayerState:
    """Public information about one player, including remembered market pickups.

    ``known_hand_counts`` contains (company ID, count) pairs ordered from 5 to
    10, including zero counts. Each count is a lower bound on that company's
    cards still in this player's hand: zero means none are publicly guaranteed,
    not that the player has none. For example, (7, 2) guarantees at least two 7s.

    Counts start at zero, increase on face-up market pickups, and decrease on
    matching investments or market offers, never below zero. Private deck draws
    do not change them. The sum of known counts cannot exceed ``hand_size``.
    This records public action history, without additional card-counting deductions.

    A tuple of pairs keeps the snapshot immutable; callers can use
    ``dict(state.known_hand_counts)`` for a separate dictionary keyed by company.
    ``coins`` is a signed balance and may be negative during play or after scoring.
    """

    player_id: PlayerId
    coins: int
    hand_size: int
    investments: tuple[CompanyId, ...]
    known_hand_counts: tuple[tuple[CompanyId, int], ...]


@dataclass(frozen=True)
class Observation:
    """An immutable snapshot of the game from one player's perspective.

    ``players`` is ordered by player ID. ``market`` preserves market indices.
    ``anti_monopoly_owners`` contains (company ID, owner ID or None) pairs.
    The viewer's hand is the only complete hand included. Each public player
    state also includes known hand counts from remembered public actions, shared
    by all observers. These do not reveal additional privately acquired cards.
    Deck order, removed card identities, opponents' complete hands, and
    random-generator state are excluded.
    """

    player_id: PlayerId
    current_player_id: PlayerId | None
    phase: Phase
    hand: tuple[CompanyId, ...]
    players: tuple[PublicPlayerState, ...]
    market: tuple[MarketCard, ...]
    deck_size: int
    anti_monopoly_owners: tuple[tuple[CompanyId, PlayerId | None], ...]
    taken_company_this_turn: CompanyId | None


@dataclass(frozen=True)
class SimulatorState:
    """Complete, immutable game state at initialization or after one step.

    Contains private information for analysis; never pass this to a Player.
    Deck order is preserved, with the last card on top. Hands, investments,
    and market entries also retain their order and duplicate cards. Mappings
    become tuples of (key, value) pairs, ordered by player or company ID;
    for example, dict(state.hands)[2] retrieves player 2's hand.

    rng_state is Random.getstate() data, suitable for Random.setstate(),
    rather than a reference to the live generator. Nested collections are
    immutable too. The history itself and fixed class constants are excluded.
    """

    rng_state: tuple[int, tuple[int, ...], float | None]
    deck: tuple[CompanyId, ...]
    removed_cards: tuple[CompanyId, ...]
    hands: tuple[tuple[PlayerId, tuple[CompanyId, ...]], ...]
    known_hand_counts: tuple[
        tuple[PlayerId, tuple[tuple[CompanyId, int], ...]], ...
    ]
    investments: tuple[tuple[PlayerId, tuple[CompanyId, ...]], ...]
    coins: tuple[tuple[PlayerId, int], ...]
    market: tuple[MarketCard, ...]
    anti_monopoly_owners: tuple[tuple[CompanyId, PlayerId | None], ...]
    current_player_id: PlayerId | None
    phase: Phase
    taken_company_this_turn: CompanyId | None
    end_pending: bool


class Simulator:
    """Own the complete state and enforce the rules of a four-player game.

    No Player instances are needed or imported. A Player is a decision-making
    policy; its cards and money belong to the game state stored here. A game
    loop passes an Observation to a Player and submits its chosen Action.

    Cards are represented by their company IDs, with repeated IDs representing
    multiple copies. For example, ``self._hands[2] == [5, 8, 8]`` means player 2
    holds one Giraffe Beer share and two Octo Coffee shares. Playing an 8 removes
    just one occurrence. IDs 1 through 4 identify players, not Player objects.

    ``_known_hand_counts[player_id][company_id]`` separately tracks the minimum
    holding guaranteed by public action history. Everyone sees the same market
    pickups and plays, so one shared record per player is sufficient. Update it
    on actions even if nobody requests an observation. Never fill these counts
    from private hands: initial cards and private draws provide no public knowledge.

    Coin balances may go negative during play. Drawing from the deck always
    charges the full market fee, with Anti-Monopoly exemptions, but affordability
    never determines whether the draw is legal. Negative balances carry into scoring.

    Construction and reset initialize the state described by the annotations.
    Observations expose each player's view and legal actions list their choices.
    Steps execute moves and settle balances when the final turn finishes.
    Final scores and all tied winners are available after settlement.
    History contains the initial state followed by one immutable state per
    successful step, including final scoring. Only reset starts a new history.
    """

    PLAYER_IDS: tuple[PlayerId, ...] = (1, 2, 3, 4)
    COMPANY_IDS: tuple[CompanyId, ...] = (5, 6, 7, 8, 9, 10)
    INITIAL_COINS = 15
    INITIAL_HAND_SIZE = 3
    REMOVED_CARD_COUNT = 5

    # Private, complete state. The deck's last element is its top card.
    _rng: Random
    _deck: list[CompanyId]
    _removed_cards: list[CompanyId]
    _hands: dict[PlayerId, list[CompanyId]]

    # State that is visible to every player through an observation.
    _known_hand_counts: dict[PlayerId, dict[CompanyId, int]]
    _investments: dict[PlayerId, list[CompanyId]]
    _coins: dict[PlayerId, int]
    _market: list[MarketCard]
    _anti_monopoly_owners: dict[CompanyId, PlayerId | None]

    # Turn bookkeeping, including the restriction imposed by taking a market card.
    _current_player_id: PlayerId | None
    _phase: Phase
    _taken_company_this_turn: CompanyId | None
    _end_pending: bool

    # Full-information snapshots, oldest first; not part of a snapshot itself.
    _history: list[SimulatorState]

    def __init__(
        self, seed: int | None = None, starting_player_id: PlayerId = 1
    ) -> None:
        """Create a private random generator and initialize a game via reset.

        Use an explicit starting player because simulated players have no ages.
        Default to player 1. Identical seeds and action sequences should produce
        identical games. Construction requires no Player objects.
        """
        self._rng = Random(seed)
        self.reset(starting_player_id=starting_player_id)

    def reset(
        self, seed: int | None = None, starting_player_id: PlayerId = 1
    ) -> None:
        """Replace all game state with a fresh four-player game.

        Reseed the private generator if a seed is provided; otherwise continue
        its existing sequence. Build 45 cards, using each company ID as its copy
        count, shuffle, remove five unseen cards, and deal three to each player.
        The draw pile then contains 28 cards. Initialize a separate hand and
        investment list per player, 15 coins each, an empty market, and unclaimed
        Anti-Monopoly chips. Give every player a separate known-hand dictionary
        with all six company counts set to zero, regardless of their actual hand.
        Clear any knowledge from the previous game. Begin in the take phase with
        no pending end or market-company restriction. Reject an invalid starting
        player ID before changing state or reseeding the generator.
        Replace history with a single snapshot of the new initial state.
        """
        if type(starting_player_id) is not int or starting_player_id not in self.PLAYER_IDS:
            raise ValueError("starting_player_id must be an integer from 1 through 4")

        if seed is not None:
            self._rng.seed(seed)

        self._deck = [
            company_id
            for company_id in self.COMPANY_IDS
            for _ in range(company_id)
        ]
        self._rng.shuffle(self._deck)
        self._removed_cards = [self._deck.pop() for _ in range(self.REMOVED_CARD_COUNT)]
        self._hands = {
            player_id: [self._deck.pop() for _ in range(self.INITIAL_HAND_SIZE)]
            for player_id in self.PLAYER_IDS
        }
        self._investments = {player_id: [] for player_id in self.PLAYER_IDS}
        self._coins = {player_id: self.INITIAL_COINS for player_id in self.PLAYER_IDS}
        self._known_hand_counts = {
            player_id: {company_id: 0 for company_id in self.COMPANY_IDS}
            for player_id in self.PLAYER_IDS
        }
        self._market = []
        self._anti_monopoly_owners = {company_id: None for company_id in self.COMPANY_IDS}
        self._current_player_id = starting_player_id
        self._phase = "take"
        self._taken_company_this_turn = None
        self._end_pending = False
        self._history = [self._snapshot_state()]

    @property
    def current_player_id(self) -> PlayerId | None:
        """Return the player whose decision is required, or None after scoring."""
        return self._current_player_id

    @property
    def is_terminal(self) -> bool:
        """Return whether the game has finished and final scores are available."""
        return self._phase == "finished"

    def get_observation(self, player_id: PlayerId) -> Observation:
        """Build an independent snapshot containing only this player's knowledge.

        Any valid player may observe, including outside their turn or after the
        game. Convert internal collections into immutable snapshot values; never
        expose mutable state. Include each player's known hand counts as six
        (company ID, count) pairs ordered by company ID, copied from the public
        knowledge record. Observation requests must not change those counts.
        Include the viewer's exact hand separately, without substituting it into
        the public counts. At game end all hand cards become public investments
        and known hand counts are zero, but the five removed cards remain hidden.
        Reject IDs outside 1 through 4.
        """
        if type(player_id) is not int or player_id not in self.PLAYER_IDS:
            raise ValueError("player_id must be an integer from 1 through 4")

        return self._observation_from_state(self._snapshot_state(), player_id)

    def get_history_state(self, turns: int | None = None) -> tuple[SimulatorState, ...]:
        """Return the most recent recorded states, ordered oldest to newest.

        turns counts snapshots, not complete two-decision player turns: 1
        returns only the latest snapshot, 0 returns (), and None returns the
        entire history. A count larger than the history returns all entries.
        Reject negative counts, booleans, and other noninteger values with
        ValueError. The initial snapshot precedes every successful step's
        snapshot; the final entry after game end includes revelation and scoring.

        Both the returned tuple and its states are immutable. This method
        exposes all hidden information and is for trusted analysis code.
        Give players get_history_observation results instead. Reading history
        never records a state, changes the live game, or consumes randomness.
        """
        if turns is None:
            return tuple(self._history)
        if type(turns) is not int or turns < 0:
            raise ValueError("turns must be a nonnegative integer or None")
        if turns == 0:
            return ()
        return tuple(self._history[-turns:])

    def get_history_observation(
        self, turns: int | None, player_id: PlayerId
    ) -> tuple[Observation, ...]:
        """Return this player's views of the most recent recorded states.

        Use the same snapshot count and oldest-to-newest order as
        get_history_state: None means all, 0 means none, and oversized counts
        return all available entries. Reject invalid counts or player IDs
        with ValueError, including invalid IDs when requesting zero entries.

        Each view reflects only information available to this player at that
        historical moment. Later pickups and hand revelations never backfill
        earlier knowledge. Do not expose other hands, deck order, removed
        cards, RNG state, or a reference to the underlying SimulatorState.
        Any valid player may request history outside their turn or after the
        game. Return immutable observations without rewinding the live game.
        """
        if type(player_id) is not int or player_id not in self.PLAYER_IDS:
            raise ValueError("player_id must be an integer from 1 through 4")

        return tuple(
            self._observation_from_state(state, player_id)
            for state in self.get_history_state(turns)
        )

    def _snapshot_state(self) -> SimulatorState:
        """Copy all per-game fields into immutable values without recording them.

        Capture RNG state without advancing it; never retain mutable lists,
        dictionaries, or the history container. Calling this helper alone
        does not append history, so observation queries remain read-only.
        """
        return SimulatorState(
            rng_state=self._rng.getstate(),
            deck=tuple(self._deck),
            removed_cards=tuple(self._removed_cards),
            hands=tuple(
                (player_id, tuple(self._hands[player_id]))
                for player_id in self.PLAYER_IDS
            ),
            known_hand_counts=tuple(
                (player_id, tuple(
                    (company_id, self._known_hand_counts[player_id][company_id])
                    for company_id in self.COMPANY_IDS
                ))
                for player_id in self.PLAYER_IDS
            ),
            investments=tuple(
                (player_id, tuple(self._investments[player_id]))
                for player_id in self.PLAYER_IDS
            ),
            coins=tuple((player_id, self._coins[player_id]) for player_id in self.PLAYER_IDS),
            market=tuple(self._market),
            anti_monopoly_owners=tuple(
                (company_id, self._anti_monopoly_owners[company_id])
                for company_id in self.COMPANY_IDS
            ),
            current_player_id=self._current_player_id,
            phase=self._phase,
            taken_company_this_turn=self._taken_company_this_turn,
            end_pending=self._end_pending,
        )

    @staticmethod
    def _observation_from_state(state: SimulatorState, player_id: PlayerId) -> Observation:
        """Filter a snapshot for an already validated viewer, using no live state.

        Share only immutable, visible values. The viewer's exact hand remains
        separate from the public known counts, even for the viewer's own seat.
        Used for both current and historical observations to keep visibility
        rules consistent. No full-state reference is included in the result.
        """
        hands = dict(state.hands)
        investments = dict(state.investments)
        coins = dict(state.coins)
        known_hand_counts = dict(state.known_hand_counts)
        return Observation(
            player_id=player_id,
            current_player_id=state.current_player_id,
            phase=state.phase,
            hand=hands[player_id],
            players=tuple(
                PublicPlayerState(
                    player_id=public_player_id,
                    coins=coins[public_player_id],
                    hand_size=len(hand),
                    investments=investments[public_player_id],
                    known_hand_counts=known_hand_counts[public_player_id],
                )
                for public_player_id, hand in state.hands
            ),
            market=state.market,
            deck_size=len(state.deck),
            anti_monopoly_owners=state.anti_monopoly_owners,
            taken_company_this_turn=state.taken_company_this_turn,
        )

    def get_legal_actions(self, player_id: PlayerId) -> tuple[Action, ...]:
        """List the legal choices for this player in the current phase.

        Return an empty tuple outside the player's turn or after the game; reject
        invalid player IDs. Account for chip ownership, cards in hand, and the
        company taken from the market this turn. During the active player's take
        phase, include draw_deck whenever the deck is nonempty, regardless of
        their coin balance or the market fee. Offer one hand action per distinct
        company, but one take action per eligible market entry.

        Order choices as draw, market pickups by index, investments by company,
        then market offers by company. Read only information available in this
        player's observation, and do not mutate state or consume randomness.
        Numerical action IDs and model masks belong to the separate ActionCodec.
        """
        if type(player_id) is not int or player_id not in self.PLAYER_IDS:
            raise ValueError("player_id must be an integer from 1 through 4")

        if self.is_terminal or player_id != self.current_player_id:
            return ()

        if self._phase == "take":
            actions = [Action("draw_deck")] if self._deck else []
            actions.extend(
                Action("take_market", market_index=market_index)
                for market_index, card in enumerate(self._market)
                if self._anti_monopoly_owners[card.company_id] != player_id
            )
            return tuple(actions)

        if self._phase == "play":
            held_companies = tuple(
                company_id
                for company_id in self.COMPANY_IDS
                if company_id in self._hands[player_id]
            )
            actions = [Action("invest", company_id=company_id) for company_id in held_companies]
            actions.extend(
                Action("offer_to_market", company_id=company_id)
                for company_id in held_companies
                if company_id != self._taken_company_this_turn
            )
            return tuple(actions)

        raise RuntimeError(f"Unknown game phase: {self._phase!r}")

    def step(self, player_id: PlayerId, action: Action) -> None:
        """Validate and execute one decision, not an entire two-decision turn.

        A take action keeps the same player active and enters the play phase.
        A play action advances clockwise, unless the last deck card was drawn:
        then finish and score the game instead. Clear turn-specific restrictions
        when advancing, but retain known hand counts across turns. Raise ValueError
        for an invalid move before any mutation, including public knowledge updates.
        Append exactly one history snapshot after all effects of a successful
        decision, including turn advancement or final revelation and scoring.
        Rejected actions add no history entries.
        """
        self._validate_action(player_id, action)

        if self._phase == "take":
            self._take_card(player_id, action)
            self._phase = "play"
        else:
            self._play_card(player_id, action)
            if self._end_pending:
                self._finish_game()
            else:
                next_player_index = (self.PLAYER_IDS.index(player_id) + 1) % len(self.PLAYER_IDS)
                self._current_player_id = self.PLAYER_IDS[next_player_index]
                self._phase = "take"
                self._taken_company_this_turn = None

        self._history.append(self._snapshot_state())

    def _validate_action(self, player_id: PlayerId, action: Action) -> None:
        """Check the player, phase, action arguments, and all move restrictions.

        Raise ValueError if the game is finished or the move is illegal. Do not
        reject a draw because the player has insufficient coins, including a zero
        or negative balance. Mutate no state and consume no randomness. Validate
        even if the caller previously obtained the action from get_legal_actions,
        since it may now be stale. After checking the action's argument types and
        structure, check membership in get_legal_actions rather than duplicating
        its rules. The action codec checks encoding validity, not move legality.
        """
        if type(player_id) is not int or player_id not in self.PLAYER_IDS:
            raise ValueError("player_id must be an integer from 1 through 4")
        if self.is_terminal:
            raise ValueError("Cannot act after the game has finished")
        if player_id != self.current_player_id:
            raise ValueError("It is not this player's turn")
        if not isinstance(action, Action):
            raise ValueError("action must be an Action instance")
        if type(action.kind) is not str:
            raise ValueError("action.kind must be a supported action name")

        if action.kind == "draw_deck":
            if action.company_id is not None or action.market_index is not None:
                raise ValueError("draw_deck must not have a company_id or market_index")
        elif action.kind == "take_market":
            if action.company_id is not None or type(action.market_index) is not int:
                raise ValueError("take_market requires only an integer market_index")
        elif action.kind in ("invest", "offer_to_market"):
            if action.market_index is not None or type(action.company_id) is not int:
                raise ValueError("invest and offer_to_market require only an integer company_id")
        else:
            raise ValueError(f"Unknown action kind: {action.kind!r}")

        if action not in self.get_legal_actions(player_id):
            raise ValueError("Action is not legal for this player in the current state")

    def _take_card(self, player_id: PlayerId, action: Action) -> None:
        """Execute a validated draw_deck or take_market action.

        Drawing adds one coin to each non-exempt market entry and deducts the
        full total from the player's balance before taking the top deck card.
        Apply the full fee even if the balance becomes or is already negative;
        never cap the payment at available coins or clamp the balance to zero.
        Drawing leaves known hand counts unchanged because the card is private.
        Taking a market entry adds its card to the hand, increments
        that player's known count for its company by one, collects its coins,
        and records its company for this turn's restriction.
        The hand grows from three to four. Drawing the last deck card sets
        _end_pending, but does not end the game before the play decision.
        """
        if action.kind == "draw_deck":
            fee = 0
            for market_index, card in enumerate(self._market):
                if self._anti_monopoly_owners[card.company_id] != player_id:
                    self._market[market_index] = MarketCard(card.company_id, card.coins + 1)
                    fee += 1
            self._coins[player_id] -= fee
            self._hands[player_id].append(self._deck.pop())
            self._taken_company_this_turn = None
            self._end_pending = not self._deck
        else:
            card = self._market.pop(action.market_index)
            self._hands[player_id].append(card.company_id)
            self._coins[player_id] += card.coins
            self._known_hand_counts[player_id][card.company_id] += 1
            self._taken_company_this_turn = card.company_id

    def _play_card(self, player_id: PlayerId, action: Action) -> None:
        """Execute a validated invest or offer_to_market action.

        Remove one matching card from the player's four-card hand. An investment
        enters their public holdings and updates the company's Anti-Monopoly
        owner; an offered card enters the market with zero attached coins.
        Either action reduces the player's known count for the played company
        by one, stopping at zero; other company counts are unchanged. Identical
        cards are interchangeable, so no physical card identity is tracked.
        The hand returns to three. Leave turn advancement to step.
        """
        company_id = action.company_id
        self._hands[player_id].remove(company_id)
        known_count = self._known_hand_counts[player_id][company_id]
        self._known_hand_counts[player_id][company_id] = max(0, known_count - 1)

        if action.kind == "invest":
            self._investments[player_id].append(company_id)
            self._update_anti_monopoly(company_id)
        else:
            self._market.append(MarketCard(company_id, 0))

    def _update_anti_monopoly(self, company_id: CompanyId) -> None:
        """Update a company's chip owner using only face-up investments.

        Award an unclaimed chip to the first investor. Transfer an existing chip
        only when another player has strictly more shares than its current owner.
        A tie preserves the current owner; modified final payouts do not change
        this rule. Hidden hand cards never influence chip ownership during play.
        """
        holdings = {
            player_id: self._investments[player_id].count(company_id)
            for player_id in self.PLAYER_IDS
        }
        leader = max(self.PLAYER_IDS, key=holdings.__getitem__)
        current_owner = self._anti_monopoly_owners[company_id]
        if holdings[leader] > 0 and (
            current_owner is None or holdings[leader] > holdings[current_owner]
        ):
            self._anti_monopoly_owners[company_id] = leader

    def _finish_game(self) -> None:
        """Reveal hands, settle all companies simultaneously, and mark completion.

        Move remaining hand cards into investments, empty every hand, and clear
        all known hand counts because no cards remain in hands. For each company,
        find all players tied for the largest positive holding. Every minority holder
        loses one coin per own share per leader; each leader gains three coins
        per minority share. Tied leaders do not pay each other. Ignore market
        cards and their coins, and do not reveal the five removed cards.

        Apply net changes without tracking chip
        denominations or limiting payments to pre-settlement balances; negative
        balances from play carry forward and final balances may also be negative.
        This avoids payout-order effects. Finish in the finished phase with no
        active player. Reject settlement without a pending end or after scoring.
        """
        if self.is_terminal:
            raise ValueError("The game has already been scored")
        if not self._end_pending or self._deck:
            raise ValueError("The final turn must finish before scoring")

        for player_id in self.PLAYER_IDS:
            self._investments[player_id].extend(self._hands[player_id])
            self._hands[player_id].clear()
            self._known_hand_counts[player_id] = {
                company_id: 0 for company_id in self.COMPANY_IDS
            }

        changes = {player_id: 0 for player_id in self.PLAYER_IDS}
        for company_id in self.COMPANY_IDS:
            holdings = {
                player_id: self._investments[player_id].count(company_id)
                for player_id in self.PLAYER_IDS
            }
            largest_holding = max(holdings.values())
            if largest_holding == 0:
                continue
            leaders = tuple(
                player_id for player_id in self.PLAYER_IDS
                if holdings[player_id] == largest_holding
            )
            for player_id, shares in holdings.items():
                if shares < largest_holding:
                    changes[player_id] -= shares * len(leaders)
                    for leader in leaders:
                        changes[leader] += shares * 3

        for player_id in self.PLAYER_IDS:
            self._coins[player_id] += changes[player_id]
        self._phase = "finished"
        self._current_player_id = None
        self._taken_company_this_turn = None
        self._end_pending = False

    def get_scores(self) -> dict[PlayerId, int]:
        """Return a fresh mapping of all four final coin balances.

        Raise ValueError before the game finishes. Scores are final balances,
        not an RL reward calculation; a training adapter can define rewards.
        """
        if not self.is_terminal:
            raise ValueError("Final scores are only available after the game has finished")
        return {player_id: self._coins[player_id] for player_id in self.PLAYER_IDS}

    def get_winner_ids(self) -> tuple[PlayerId, ...]:
        """Return all players with the highest final balance, ordered by ID.

        Share wins when balances are equal, since numeric
        balances have no 3-point-chip count to use as a tiebreaker. Raise ValueError
        before the game finishes.
        """
        scores = self.get_scores()
        highest_score = max(scores.values())
        return tuple(
            player_id for player_id in self.PLAYER_IDS
            if scores[player_id] == highest_score
        )
