"""Handwritten strategies and baselines using the shared RuleBasedPlayer wrapper.

Add each distinct strategy in its own module. Policies can subclass
StructuredRuleBasedPlayer and fill its four decision hooks, or subclass
RuleBasedPlayer and supply policy directly when a different decomposition fits.
Variants that differ only in thresholds or weights can use configuration on
the same class. Internal modules import their base directly from its module,
rather than through the top-level players package.

RandomPlayer is a working uniform baseline using the unrestricted wrapper.
RuleBasedPlayer supplies shared counting and public token prediction.
StructuredRuleBasedPlayer adds legality filtering and phase dispatch.
ThresholdPlayer implements the collection/threshold strategy. THRESHOLD_CONFIGS
holds its five parameter profiles; create_threshold_player applies any profile
with optional chicken mode, avoiding new tokens except when forced to invest.
RankingPlayer ranks legal actions and samples exact best-score ties with a private
RNG. WeightedRankingPlayer supplies shared features through RankingWeights and
exposes immutable ScoredAction explanations. Named dictionary configurations in
RANKING_CONFIGS supply numbered weight profiles without new classes.
create_ranking_player constructs one configuration with an independent RNG.
"""

from .base import RuleBasedPlayer
from .random_player import RandomPlayer
from .ranking_base import RankingPlayer, ScoredAction
from .weighted_ranking_player import WeightedRankingPlayer, RankingWeights
from .ranking_configs import RANKING_CONFIGS, create_ranking_player
from .structured_base import PlayDecision, StructuredRuleBasedPlayer
from .threshold_player import ThresholdPlayer, ThresholdParameters
from .threshold_configs import THRESHOLD_CONFIGS, create_threshold_player


__all__ = [
    "PlayDecision", "RandomPlayer", "RuleBasedPlayer", "StructuredRuleBasedPlayer",
    "RankingPlayer", "WeightedRankingPlayer", "RankingWeights", "ScoredAction",
    "RANKING_CONFIGS", "create_ranking_player",
    "ThresholdPlayer", "ThresholdParameters", "THRESHOLD_CONFIGS", "create_threshold_player",
]
