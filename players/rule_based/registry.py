"""Named opponent specifications reusable by the CLI and evaluation/training loops.

Entries identify strategies/configurations rather than unique Python classes.
Uniform sampling over entries gives every named configuration the same chance;
sampling with replacement is allowed. Controllers are constructed only after
sampling, with independent seats and RNG seeds. Configuration dictionaries are
validated and frozen into the registry on import, not read during gameplay.
"""

from dataclasses import dataclass, fields

from simulator import PlayerId

from .base import RuleBasedPlayer
from .random_player import RandomPlayer
from .ranking_base import RankingPlayer
from .ranking_configs import RANKING_CONFIGS
from .weighted_ranking_player import RankingWeights, WeightedRankingPlayer
from .threshold_player import ThresholdPlayer, ThresholdParameters
from .threshold_configs import THRESHOLD_CONFIGS


@dataclass(frozen=True)
class OpponentSpec:
    """A named strategy and frozen settings, not a live player instance."""

    name: str
    player_class: type[RuleBasedPlayer]
    weights: RankingWeights | None = None
    parameters: ThresholdParameters | None = None
    chicken: bool = False

    def create(self, player_id: PlayerId, *, seed: int | None = None) -> RuleBasedPlayer:
        """Construct a fresh controller with this name and an independent RNG.

        Weighted evaluators receive this entry's frozen weights. Other stochastic
        players receive the seed; deterministic threshold strategies need none.
        Reusing an entry never reuses a controller or its random generator.
        """
        if issubclass(self.player_class, ThresholdPlayer):
            if self.weights is not None:
                raise ValueError("Threshold players do not accept ranking weights")
            return self.player_class(player_id, self.name, parameters=self.parameters, chicken=self.chicken)
        if self.parameters is not None or self.chicken:
            raise ValueError("Only threshold players accept threshold parameters/chicken mode")
        if issubclass(self.player_class, WeightedRankingPlayer):
            return self.player_class(player_id, self.name, seed=seed, weights=self.weights)
        if self.weights is not None:
            raise ValueError("Only weighted ranking players accept registry weights")
        if issubclass(self.player_class, (RandomPlayer, RankingPlayer)):
            return self.player_class(player_id, self.name, seed=seed)
        return self.player_class(player_id, self.name)

    @property
    def description(self) -> str:
        """Describe the strategy, including nondefault ranking weight overrides."""
        if issubclass(self.player_class, WeightedRankingPlayer):
            defaults = self.player_class.DEFAULT_WEIGHTS
            weights = defaults if self.weights is None else self.weights
            changes = ", ".join(
                f"{field.name}={getattr(weights, field.name)}"
                for field in fields(weights)
                if getattr(weights, field.name) != getattr(defaults, field.name)
            )
            return (
                "ranks resulting positions + action events; uniformly random best-score ties; "
                + (changes or "standard weights")
            )
        if issubclass(self.player_class, RandomPlayer):
            return "uniformly random legal actions"
        parameters = ThresholdParameters() if self.parameters is None else self.parameters
        detail = (
            f"pickup: hand >= {parameters.hand_pickup_threshold} OR "
            f"hand + investments >= {parameters.own_pickup_threshold}; "
            f"coin fallback >= {parameters.coin_pickup_threshold}; "
            f"invest: i // {parameters.investment_divisor}"
        )
        if self.chicken:
            detail += "; avoid new tokens unless forced to invest"
        return detail


RULE_BASED_OPPONENTS: tuple[OpponentSpec, ...] = (OpponentSpec("RandomPlayer", RandomPlayer),) + tuple(
    OpponentSpec(
        f"Threshold{'Chicken' if chicken else 'Player'}{profile}", ThresholdPlayer,
        parameters=ThresholdParameters(**overrides), chicken=chicken,
    )
    for chicken in (False, True)
    for profile, overrides in THRESHOLD_CONFIGS.items()
) + tuple(
    OpponentSpec(f"Ranking ({name})", WeightedRankingPlayer, RankingWeights(**overrides))
    for name, overrides in RANKING_CONFIGS.items()
)
