"""A uniformly random legal-action baseline with an independent generator."""

from random import Random

from simulator import Observation, PlayerId

from ..base import ActionMask
from .base import RuleBasedPlayer


class RandomPlayer(RuleBasedPlayer):
    """Give each enabled action ID equal probability on every decision.

    Each market slot is a separate action. Investment and offer actions are per
    company, so holding duplicate cards does not increase their probability.
    This stochastic baseline shares the rule-based decision wrapper but keeps
    its own RNG, independent of other players and the simulator's deck shuffle.
    The same seed and sequence of masks reproduce the same sequence of choices.
    """

    _rng: Random

    def __init__(
        self, player_id: PlayerId, name: str = "Random player", *, seed: int | None = None
    ) -> None:
        """Initialize identity and a private RNG, optionally seeded by an integer.

        Reject noninteger seeds, including booleans, with ValueError. With None,
        let Random initialize from system randomness. Do not consume or reseed
        the process-wide RNG or create any game state.
        """
        super().__init__(player_id, name)
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        self._rng = Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Preserve identity and continue the RNG unless an integer seed is supplied.

        An explicit seed restarts the sequence for reproducibility. There is no
        retained observation context. Reject an invalid seed before changing RNG
        state; resetting this player never affects another player's generator.
        """
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        super().reset(seed)
        if seed is not None:
            self._rng.seed(seed)

    def policy(self, observation: Observation, mask: ActionMask) -> int:
        """Validate the request and sample uniformly from the enabled action IDs.

        Validation also protects direct policy calls: invalid requests consume
        no randomness. Beyond the validation metadata, the observation does not
        influence the choice. Read no simulator state and retain no observations.
        """
        self._validate_request(observation, mask)
        legal_ids = tuple(action_id for action_id, allowed in enumerate(mask) if allowed)
        return self._rng.choice(legal_ids)
