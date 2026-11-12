"""Framework-independent controller for masked, epsilon-greedy Q-value policies.

The model adapter converts the encoder's result into 41 numerical action values.
Architecture, tensor conversion, weight loading and optimization stay outside
the controller. Importing this module does not require PyTorch.
"""

from collections.abc import Callable, Sequence
from math import isfinite
from numbers import Real
from random import Random
from typing import Literal

from action_codec import ActionCodec
from simulator import Observation, PlayerId

from ..base import ActionMask, Player, PlayerKind


SelectionMode = Literal["epsilon_greedy", "greedy"]
ObservationEncoder = Callable[[Observation], object]
QValueModel = Callable[[object], Sequence[float]]


class NeuralPlayer(Player):
    """Select legal actions using Q-values, with optional uniform exploration.

    Pass QNetwork.predict as the model adapter and FlatObservationEncoder() as
    the encoder for the first model. Q-values are not logits or probabilities:
    no softmax is used. The legality mask is applied during selection and is
    never added to the model input. Greedy ties use the lowest action ID.

    Model parameters may be shared between controllers; each controller owns
    its exploration RNG. This first policy uses one observation and retains no
    history. Future historical/recurrent context must be isolated per player.
    Loading weights and training remain external; checkpoint_id is log metadata.
    """

    def __init__(
        self,
        player_id: PlayerId,
        model: QValueModel,
        observation_encoder: ObservationEncoder,
        *,
        name: str = "Neural player",
        selection_mode: SelectionMode = "greedy",
        epsilon: float = 0.0,
        seed: int | None = None,
        checkpoint_id: str | None = None,
    ) -> None:
        """Validate adapters and settings, then create an independent sampler.

        Greedy mode ignores epsilon. Epsilon-greedy mode explores with probability
        epsilon, uniformly across legal actions (including the greedy action).
        A future trainer owns the schedule and can assign player.epsilon between
        games. This constructor does not load weights or initialize an optimizer.
        """
        super().__init__(player_id, name)
        if not callable(model) or not callable(observation_encoder):
            raise ValueError("model and observation_encoder must be callable")
        if selection_mode not in ("greedy", "epsilon_greedy"):
            raise ValueError("selection_mode must be greedy or epsilon_greedy")
        if checkpoint_id is not None and (
            not isinstance(checkpoint_id, str) or not checkpoint_id.strip()
        ):
            raise ValueError("checkpoint_id must be a nonempty string or None")
        self.model = model
        self.observation_encoder = observation_encoder
        self.selection_mode = selection_mode
        self.epsilon = epsilon
        self.checkpoint_id = checkpoint_id
        self._rng = Random(seed)

    @property
    def kind(self) -> PlayerKind:
        """Identify this player as a neural-policy controller."""
        return "neural"

    @property
    def epsilon(self) -> float:
        """Return the exploration probability, separate from model parameters."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set a finite exploration probability in [0, 1], rejecting booleans."""
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
            raise ValueError("epsilon must be a finite real number between 0 and 1")
        self._epsilon = float(value)

    def reset(self, seed: int | None = None) -> None:
        """Reseed only when requested, preserving the model and current settings.

        No observations are retained by this first controller. With seed=None,
        continue the exploration stream across games rather than repeating it.
        Future per-game context belongs here; shared weights must remain intact.
        """
        if seed is not None:
            self._rng.seed(seed)

    async def choose_action(self, observation: Observation, mask: ActionMask) -> int:
        """Validate a decision, call policy, and validate its chosen action ID.

        Neither advance the simulator nor implicitly call on_observation. The
        game loop remains responsible for observations and applying the action.
        """
        self._validate_request(observation, mask)
        return self._validate_action_id(self.policy(observation, mask), mask)

    def policy(self, observation: Observation, mask: ActionMask) -> int:
        """Select a legal action from an already validated decision request.

        Exploration skips model evaluation entirely. Exploitation encodes the
        current observation once, requires 41 values, and compares only legal
        entries. Legal Q-values must be finite real numbers. Masked values are
        ignored, even if nonfinite. Greedy selection consumes no random numbers.

        The adapter owns inference execution; this method never changes weights,
        train/eval mode or gradients. Use choose_action for request validation.
        """
        legal_ids = tuple(i for i, allowed in enumerate(mask) if allowed)
        if self.selection_mode == "epsilon_greedy" and self.epsilon > 0:
            if self.epsilon == 1 or self._rng.random() < self.epsilon:
                return self._rng.choice(legal_ids)

        encoded = self.observation_encoder(observation)
        output = self.model(encoded)
        try:
            values = tuple(output)
        except TypeError as error:
            raise ValueError("model must return a sequence of 41 Q-values") from error
        if len(values) != ActionCodec.ACTION_COUNT:
            raise ValueError("model must return exactly 41 Q-values")
        for action_id in legal_ids:
            value = values[action_id]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("legal Q-values must be finite real numbers")
            try:
                finite = isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("legal Q-values must be finite real numbers")
        return max(legal_ids, key=values.__getitem__)
