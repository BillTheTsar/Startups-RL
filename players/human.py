"""Human controller with a numbered CLI and cancellable input requests."""

from asyncio import CancelledError, Task, create_task

from action_codec import ActionCodec
from simulator import Observation, PlayerId

from .base import ActionMask, Player, PlayerKind
from .cli_input import console_session, read_line


class HumanPlayer(Player):
    """Display the player's view and wait for a legal choice from a person.

    The game loop awaits CLI input through choose_action before advancing this
    game; other async tasks may continue. Each player can have only one pending
    decision. Reset cancels that decision so a response from the previous game
    cannot become a new move. The player receives only its own observation and
    mask, with no reference to the simulator.
    """

    _pending_request: Task[int] | None

    def __init__(
        self,
        player_id: PlayerId,
        name: str = "Human player",
    ) -> None:
        """Validate the common identity and start with no pending CLI decision.

        No event loop, console read, or simulator is created during construction.
        The CLI is the input interface; a graphical adapter can be added later.
        """
        super().__init__(player_id, name)
        self._pending_request = None

    @property
    def kind(self) -> PlayerKind:
        """Identify this player as a human-controlled seat."""
        return "human"

    def reset(self, seed: int | None = None) -> None:
        """Cancel a pending async decision and ignore its eventual stale response.

        Preserve identity and ignore seed, as human choices have no sampler.
        Invalidate the request before cancelling it; choose_action checks request
        identity too, so even a late result cannot be returned as a valid move.
        Repeated resets are harmless. No simulator state is changed.
        """
        super().reset(seed)
        request = self._pending_request
        self._pending_request = None
        if request is not None and not request.done():
            request.cancel()

    async def choose_action(self, observation: Observation, mask: ActionMask) -> int:
        """Validate, display the supplied view, and await one human action ID.

        Reject overlapping decisions on this player with RuntimeError. Validate
        the request before opening the CLI, and validate the returned action ID
        against the same mask. Release the request on success, cancellation, EOF,
        or error. Reset invalidates an in-flight result even if it arrives late.
        Do not invoke on_observation or advance a simulator here; the game loop
        submits the returned choice. Cancellation and EOF propagate to that loop.
        """
        self._validate_request(observation, mask)
        if self._pending_request is not None:
            raise RuntimeError("This player already has a pending decision")

        request = create_task(self._choose_from_cli(observation, mask))
        self._pending_request = request
        try:
            action_id = await request
            if self._pending_request is not request:
                raise CancelledError("The human decision was reset")
            return self._validate_action_id(action_id, mask)
        finally:
            if self._pending_request is request:
                self._pending_request = None

    def _format_observation(self, observation: Observation) -> str:
        """Build a readable command-line view from only this observation.

        Use labeled text sections with one player's public information at a time.
        Show the viewer's exact hand separately. Show investment and known-hand
        counts for every company, preserving zeros. Known counts are guarantees,
        not complete opponent hands. Market cards use human-facing numbers
        starting at 1, in observation order, with their companies and coins.
        Include turn, deck size, every chip owner, and the offer restriction.
        No hidden cards or RNG data are available through this interface.
        """
        self._validate_observation(observation)
        companies = tuple(company for company, _ in observation.anti_monopoly_owners)
        active_player = observation.current_player_id
        lines = [
            f"\n=== {self.name} (Player {self.player_id}) ===",
            "Turn",
            f"  Phase: {observation.phase}",
            f"  Active player: {active_player if active_player is not None else 'none'}",
            f"  Cards remaining in deck: {observation.deck_size}",
            "",
            "Your hand",
            "  " + (", ".join(str(company) for company in observation.hand) or "(empty)"),
        ]
        for player in observation.players:
            investments = ", ".join(
                f"{company}: {player.investments.count(company)}" for company in companies
            )
            known_cards = ", ".join(
                f"{company}: {count}" for company, count in player.known_hand_counts
            )
            lines.extend([
                "",
                f"Player {player.player_id}" + (" (you)" if player.player_id == self.player_id else ""),
                f"  Coins: {player.coins}",
                f"  Cards in hand: {player.hand_size}",
                f"  Investments (company: count): {investments}",
                f"  Publicly known hand minimums (company: count): {known_cards}",
            ])
        lines.extend([
            "  Known minimums record public pickups; 0 means no card is guaranteed.",
            "",
            "Anti-Monopoly chips",
        ])
        lines.extend(
            f"  Company {company}: " + (f"Player {owner}" if owner is not None else "unclaimed")
            for company, owner in observation.anti_monopoly_owners
        )
        lines.extend(["", "Market"])
        if observation.market:
            lines.extend(
                f"  Card #{number}: company {card.company_id}, attached coins: {card.coins}"
                for number, card in enumerate(observation.market, start=1)
            )
        else:
            lines.append("  (empty)")
        lines.extend(["", "Current-turn restriction"])
        taken_company = observation.taken_company_this_turn
        lines.append(
            "  None."
            if taken_company is None
            else f"  Took company {taken_company} from the market; cannot offer company {taken_company} this turn."
        )
        return "\n".join(lines)

    def _format_legal_actions(self, mask: ActionMask) -> str:
        """List legal choices as a consecutive menu numbered from 1.

        Order choices by their fixed ActionCodec ID but display only menu
        numbers. Market card numbers also start at 1 and match the market list.
        The CLI translates the selected menu number back to the fixed action ID;
        menu numbers can change between decisions and are not model action IDs.
        """
        self._validate_mask(mask)
        lines = ["\nLegal choices"]
        action_ids = tuple(action_id for action_id, allowed in enumerate(mask) if allowed)
        for number, action_id in enumerate(action_ids, start=1):
            action = ActionCodec.decode(action_id)
            if action.kind == "draw_deck":
                description = "Draw a card from the deck"
            elif action.kind == "take_market":
                description = f"Take market card #{action.market_index + 1}"
            elif action.kind == "invest":
                description = f"Invest one company {action.company_id} card"
            else:
                description = f"Offer one company {action.company_id} card to the market"
            lines.append(f"  {number}. {description}")
        return "\n".join(lines)

    async def _choose_from_cli(self, observation: Observation, mask: ActionMask) -> int:
        """Display the observation and legal choices, then await terminal input.

        Show the full view in labeled sections and a consecutive numbered menu.
        Accept a menu number with surrounding whitespace. Wait for Enter before
        submitting it, with no separate confirmation prompt. Re-prompt for
        malformed or out-of-menu choices.
        Return the corresponding fixed ActionCodec ID, not the menu number.

        The prompt accepts numeric choices only; Ctrl+C or end-of-input stops
        the decision through cancellation or EOFError. Reserve the console for
        this whole decision, rejecting concurrent human prompts. Cooperative
        input reads leave no blocking worker behind on cancellation. Do not
        call on_observation, retain history, or advance a simulator.
        """
        self._validate_request(observation, mask)
        action_ids = tuple(action_id for action_id, allowed in enumerate(mask) if allowed)
        with console_session():
            print(self._format_observation(observation))
            print(self._format_legal_actions(mask))
            print("Type a menu number, then press Enter to submit. Use Ctrl+C or end-of-input to stop.")
            while True:
                text = (await read_line(f"Choose [1-{len(action_ids)}]: ")).strip()
                if not text.isascii() or not text.isdecimal():
                    print(f"Please enter a whole menu number from 1 to {len(action_ids)}.")
                    continue
                try:
                    choice = int(text)
                except ValueError:
                    print(f"Please enter a whole menu number from 1 to {len(action_ids)}.")
                    continue
                if not 1 <= choice <= len(action_ids):
                    print(f"That choice is not available. Enter a number from 1 to {len(action_ids)}.")
                    continue
                return self._validate_action_id(action_ids[choice - 1], mask)
