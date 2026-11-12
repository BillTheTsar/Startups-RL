"""Build transitions between one player's actual decision points.

Taking leads to that player's play decision; playing usually leads to their
next take decision, after the other players have acted. A terminal observation
closes the pending action instead. An action-limit ending also closes it, using
the player's last actual live view and a terminal flag without inventing normal
settlement. No hidden state or simulated action replay is used. This module
neither runs games nor stores a replay buffer.
"""

from array import array
from dataclasses import dataclass
from math import isfinite
from typing import SupportsIndex

from action_codec import ActionCodec
from players.base import ActionMask
from players.neural.encoder import FlatObservationEncoder
from simulator import Observation, PlayerId

from .rewards import TIMEOUT_REWARD


EncodedState = tuple[float, ...]
TERMINAL_MASK: ActionMask = (False,) * ActionCodec.ACTION_COUNT


def _state(values) -> EncodedState:
    """Own an immutable, finite float32 vector; reuse already canonical tuples."""
    try:
        values = tuple(values)
        if len(values) != FlatObservationEncoder.SIZE:
            raise ValueError("state must have 332 entries")
        if any(isinstance(value, bool) or not isfinite(value) for value in values):
            raise ValueError("state entries must be finite numbers")
        floats = array("f", values)
        if floats.itemsize != 4 or any(not isfinite(value) for value in floats):
            raise ValueError("state must be representable as finite float32 values")
        normalized = tuple(floats)
    except (TypeError, OverflowError) as error:
        raise ValueError("state must contain finite float32 numbers") from error
    # Adjacent transitions can share their immutable connecting state.
    if all(type(value) is float for value in values) and normalized == values:
        return values
    return normalized


def _mask(values, *, terminal: bool = False) -> ActionMask:
    """Copy a mask and require legal slots, or all false for a terminal view."""
    try:
        mask = tuple(values)
    except TypeError as error:
        raise ValueError("mask must contain 41 booleans") from error
    if len(mask) != ActionCodec.ACTION_COUNT or any(type(value) is not bool for value in mask):
        raise ValueError("mask must contain 41 booleans")
    if terminal and any(mask):
        raise ValueError("a terminal next mask must be all false")
    if not terminal and not any(mask):
        raise ValueError("a decision mask must allow at least one action")
    return mask


def _reward(value) -> float:
    """Accept fractional rewards throughout the agreed [-1.5, 1.5] range."""
    if isinstance(value, bool):
        raise ValueError("reward must be a finite number between -1.5 and 1.5")
    try:
        valid = isfinite(value) and -1.5 <= value <= 1.5
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("reward must be a finite number between -1.5 and 1.5")
    return float(value)


@dataclass(frozen=True, slots=True)
class Transition:
    """Immutable (s, a, r, s', d, m, m') using the existing encoder and codec.

    States are 332 float32 values represented as immutable Python tuples;
    masks have 41 booleans. Construction owns/copies mutable input collections.
    Intermediate rewards are zero. A terminal reward may also be zero: use
    done, never reward or the encoded phase, to decide whether bootstrapping is
    allowed. At a timeout, next_state retains the last live observation while
    done=True and next_mask is all false: no further decision follows.
    """

    state: EncodedState
    action_id: int
    reward: float
    next_state: EncodedState
    done: bool
    mask: ActionMask
    next_mask: ActionMask

    def __post_init__(self) -> None:
        if type(self.done) is not bool:
            raise ValueError("done must be a boolean")
        mask = _mask(self.mask)
        next_mask = _mask(self.next_mask, terminal=self.done)
        action_id = ActionCodec.encode(ActionCodec.decode(self.action_id))
        if not mask[action_id]:
            raise ValueError("action_id must be legal under the current mask")
        reward = _reward(self.reward)
        if not self.done and reward != 0:
            raise ValueError("intermediate rewards must be zero")
        for name, value in (
            ("state", _state(self.state)), ("next_state", _state(self.next_state)),
            ("action_id", action_id), ("reward", reward),
            ("mask", mask), ("next_mask", next_mask),
        ):
            object.__setattr__(self, name, value)


class TransitionBuilder:
    """Keep at most one pending action for a fixed viewer.

    Each begin_action returns the preceding completed transition, if any.
    Alternatively, complete_pending closes it before choosing the next action,
    allowing a learner update before that choice; begin_action then remembers
    the new choice and returns None. finish_episode returns the last transition.
    The caller decides where to store them. reset discards pending work and
    allows another episode. Use live methods only at actual decision points,
    not ordinary observation callbacks.
    """

    def __init__(self, player_id: PlayerId) -> None:
        if type(player_id) is not int or player_id not in (1, 2, 3, 4):
            raise ValueError("player_id must be an integer from 1 through 4")
        self.player_id = player_id
        self.reset()

    def reset(self) -> None:
        """Discard any unfinished action and clear the episode boundary."""
        self._pending: tuple[EncodedState, int, ActionMask] | None = None
        self._finished = False

    def _check_view(self, observation: Observation) -> None:
        if self._finished:
            raise ValueError("episode already finished; reset before starting another")
        if not isinstance(observation, Observation) or (
            type(observation.player_id) is not int or observation.player_id != self.player_id
        ):
            raise ValueError("observation must belong to the builder's player")

    def begin_action(
        self, observation: Observation, action_id: SupportsIndex, mask: ActionMask,
    ) -> Transition | None:
        """Complete the previous action at this decision and remember this choice.

        Validate before changing pending state. This method checks the interface,
        not freshness or game rules; the game loop supplies the authoritative
        mask and the Simulator validates the chosen action when applying it.
        """
        mask = self._decision_mask(observation, mask)
        action_id = ActionCodec.encode(ActionCodec.decode(action_id))
        if not mask[action_id]:
            raise ValueError("action_id must be legal under the current mask")
        state = tuple(FlatObservationEncoder.encode(observation))
        completed = self._complete(state, mask, reward=0.0, done=False)
        self._pending = (state, action_id, mask)
        return completed

    def complete_pending(self, observation: Observation, mask: ActionMask) -> Transition | None:
        """Close the previous choice at this player's next live decision.

        Return its zero-reward, nonterminal transition and clear the pending
        choice. Return None before the first choice or after an earlier call
        already completed it. This does not store a new action: the caller can
        await a training update, choose with the updated policy, and then call
        begin_action. Validate the live view and mask even when nothing is
        pending. Invalid input leaves the pending choice unchanged.
        """
        mask = self._decision_mask(observation, mask)
        if self._pending is None:
            return None
        state = tuple(FlatObservationEncoder.encode(observation))
        completed = self._complete(state, mask, reward=0.0, done=False)
        self._pending = None
        return completed

    def _decision_mask(self, observation: Observation, mask: ActionMask) -> ActionMask:
        """Validate a live owned decision without changing pending state."""
        self._check_view(observation)
        if type(observation.current_player_id) is not int or observation.current_player_id != self.player_id:
            raise ValueError("a live decision requires this player's turn")
        if type(observation.phase) is not str or observation.phase not in ("take", "play"):
            raise ValueError("a live decision requires the take or play phase")
        return _mask(mask)

    def finish_episode(self, terminal_observation: Observation, reward: float) -> Transition | None:
        """Close the last choice at the real terminal view, even after other turns.

        Return None if this player never acted. Ending with reward zero still
        produces done=True. Finishing twice is an error until reset is called.
        """
        self._check_view(terminal_observation)
        if terminal_observation.phase != "finished" or terminal_observation.current_player_id is not None:
            raise ValueError("finish_episode requires a terminal observation")
        reward = _reward(reward)
        state = tuple(FlatObservationEncoder.encode(terminal_observation))
        completed = self._complete(state, TERMINAL_MASK, reward=reward, done=True)
        self._pending = None
        self._finished = True
        return completed

    def finish_timeout(self, observation: Observation, reward: float = TIMEOUT_REWARD) -> Transition | None:
        """End the pending choice at the action limit without faking settlement.

        Encode the final actual live view, including its take/play phase and
        visible cards. Emit done=True and an empty next mask so DQN does not
        bootstrap from that view. Hidden hands remain hidden and the simulator
        itself is unchanged. Return None if this player never acted; no action
        is invented just to attach the timeout reward. Invalid input leaves the
        pending choice intact. A second finish requires reset first.
        """
        self._check_view(observation)
        if observation.phase not in ("take", "play") or (
            type(observation.current_player_id) is not int
            or observation.current_player_id not in (1, 2, 3, 4)
        ):
            raise ValueError("finish_timeout requires a live observation")
        reward = _reward(reward)
        state = tuple(FlatObservationEncoder.encode(observation))
        completed = self._complete(state, TERMINAL_MASK, reward=reward, done=True)
        self._pending = None
        self._finished = True
        return completed

    def _complete(self, next_state, next_mask, *, reward, done) -> Transition | None:
        if self._pending is None:
            return None
        state, action_id, mask = self._pending
        return Transition(state, action_id, reward, next_state, done, mask, next_mask)
