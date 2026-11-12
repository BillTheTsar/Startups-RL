"""Common identity, observation callbacks, and decision validation for players."""

from abc import ABC, abstractmethod
from typing import Literal, SupportsIndex

from action_codec import ActionCodec
from simulator import Observation, PlayerId


ActionMask = tuple[bool, ...]
PlayerKind = Literal["rule_based", "neural", "human"]


class Player(ABC):
    """Common identity and decision interface for a seat in the game.

    The game loop supplies an observation and mask from the same decision.
    Hands, balances, and public card knowledge remain owned by the Simulator.
    Subclasses determine their kind rather than accepting conflicting type flags.
    Shared checks validate the request interface, not game legality or freshness;
    the simulator revalidates the chosen action against its current state.
    Optional context derived from this player's observations belongs to that
    player and must be cleared by its reset override between games.
    """

    player_id: PlayerId
    name: str

    def __init__(self, player_id: PlayerId, name: str) -> None:
        """Validate and store an integer seat ID from 1 through 4 and a name.

        Reject booleans and other noninteger IDs. Require a string containing
        at least one non-whitespace character, preserving the supplied text.
        Raise ValueError before assigning attributes if either input is invalid.
        Do not create a game, deal cards, or hold a Simulator reference. The
        game loop is responsible for assigning distinct IDs to its four players.
        """
        if type(player_id) is not int or player_id not in (1, 2, 3, 4):
            raise ValueError("player_id must be an integer from 1 through 4")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty display string")
        self.player_id = player_id
        self.name = name

    @property
    @abstractmethod
    def kind(self) -> PlayerKind:
        """Return the subclass's fixed kind for display, logging, and evaluation."""
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> None:
        """Prepare this player for a new game without changing identity.

        The game loop calls this once per game, before delivering observations.
        The base implementation does nothing and ignores seed because it owns
        only identity. Subclasses may call super().reset(seed) from overrides.

        Overrides must clear any retained observation history, derived context,
        or recurrent model state. Stochastic players reseed only when a seed
        is provided. Human players cancel pending input and discard responses
        from the prior game. Keep model parameters and player identity intact.
        """
        return None

    def on_observation(self, observation: Observation) -> None:
        """Optionally receive an observation update without choosing an action.

        The game loop may call this once after game initialization and after each
        completed step, including other players' decisions and the terminal state.
        Always supply this player's own view, never another player's private view
        or the full Simulator state. No active-turn requirement or mask applies.

        The default validates the Observation type and viewer ID, raising
        ValueError for a mismatched view, and retains nothing. Overrides may
        call super().on_observation(observation) before logging or maintaining
        context. Clear retained per-game data in reset. A loop may also replay
        this player's historical observations in chronological order, delivering
        each snapshot once. Do not advance the game or request an action here.
        choose_action must not call this implicitly, to avoid duplicate updates.
        """
        self._validate_observation(observation)

    @abstractmethod
    async def choose_action(self, observation: Observation, mask: ActionMask) -> int:
        """Return a legal action ID from 0 through 40 for this decision.

        Validate the request, obtain a choice, and validate the returned ID.
        A True mask entry means legal. Never silently replace an invalid policy
        output with another move. Waiting for human input pauses this game until
        the loop receives a decision; it must not advance the Simulator meanwhile.
        Subclasses implement selection and call _validate_request followed by
        _validate_action_id. They may use player-visible context retained by
        on_observation, without changing this current-observation interface.
        """
        raise NotImplementedError

    def _validate_observation(self, observation: Observation) -> None:
        """Require an Observation belonging to this player's integer seat ID.

        Raise ValueError for a wrong type or viewer, including boolean or float
        IDs that compare equal to integers. Do not require an active turn, since
        the callback also receives historical, out-of-turn, and terminal views.
        This checks ownership metadata, not every game field in the observation.
        """
        if not isinstance(observation, Observation):
            raise ValueError("observation must be an Observation instance")
        if type(observation.player_id) is not int or observation.player_id != self.player_id:
            raise ValueError("observation must belong to this player")

    @staticmethod
    def _validate_mask(mask: ActionMask) -> None:
        """Require an immutable decision mask with 41 Python bools and a legal slot.

        Raise ValueError for a wrong container, size, element type, or an
        all-False mask. Numerical array/tensor conversion belongs in adapters.
        This checks the mask's structure; the simulator determines legality.
        """
        if not isinstance(mask, tuple) or len(mask) != ActionCodec.ACTION_COUNT:
            raise ValueError(f"mask must be a tuple of {ActionCodec.ACTION_COUNT} booleans")
        if any(type(allowed) is not bool for allowed in mask):
            raise ValueError("mask entries must be Python booleans")
        if not any(mask):
            raise ValueError("mask must allow at least one action")

    def _validate_request(self, observation: Observation, mask: ActionMask) -> None:
        """Check that the request is for this player and a live decision.

        Require matching observation.player_id and current_player_id, a take or
        play phase, and an immutable mask with ActionCodec.ACTION_COUNT boolean
        entries and at least one True. Reject invalid requests with ValueError.
        Do not duplicate game rules to rebuild the mask; the Simulator remains
        authoritative and revalidates the selected action when step is called.
        Read no simulator state, invoke no observation callback, and mutate no
        player data. A valid shape cannot establish that the request is fresh.
        """
        self._validate_observation(observation)
        if (
            type(observation.current_player_id) is not int
            or observation.current_player_id != self.player_id
        ):
            raise ValueError("It is not this player's turn")
        if type(observation.phase) is not str or observation.phase not in ("take", "play"):
            raise ValueError("An action requires the take or play phase")
        self._validate_mask(mask)

    def _validate_action_id(self, action_id: SupportsIndex, mask: ActionMask) -> int:
        """Normalize an integer-like selection and ensure its mask entry is True.

        Accept integer-like model outputs through the ActionCodec decoding
        contract, rejecting booleans, nonintegers, out-of-range IDs, and masked
        choices with ValueError. Also reject malformed or all-False masks.
        Decode and re-encode once to normalize via the shared codec, then check
        the mask. Return a plain Python int without changing state or replacing
        an invalid choice with another action.
        """
        self._validate_mask(mask)
        normalized_id = ActionCodec.encode(ActionCodec.decode(action_id))
        if not mask[normalized_id]:
            raise ValueError("The selected action is masked out")
        return normalized_id
