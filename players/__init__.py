"""Interfaces for handwritten strategies, random baselines, neural policies, and humans.

Players receive only their own Observation and a 41-entry action mask. They
return an action ID; the game loop decodes it with ActionCodec and calls
Simulator.step. No player stores or receives a Simulator reference.

Player implements shared identity, validation, and default lifecycle hooks.
Its kind and choose_action contracts are abstract. HumanPlayer provides CLI
input. RuleBasedPlayer implements initialization, reset, observation helpers, and decision validation;
its policy is still a strategy placeholder. RandomPlayer is a working baseline.
StructuredRuleBasedPlayer adds legal-action filtering and phase
dispatch through four abstract decision hooks. ThresholdPlayer uses five
THRESHOLD_CONFIGS profiles and optional chicken mode, which avoids acquiring
new tokens unless forced to invest. create_threshold_player constructs all ten
combinations. WeightedRankingPlayer uses tunable
immediate-value scoring and seeded random best-score ties through RankingPlayer.
RankingWeights and ScoredAction support tuning and inspection. RANKING_CONFIGS
and create_ranking_player provide named dictionary variants using one evaluator.
NeuralPlayer selects masked Q-values with greedy or epsilon-greedy behavior.
FlatObservationEncoder provides a versioned,
332-value float32 representation of one player's observation.

Simulator owns the complete snapshot history. game_loop.run_game delivers each
player's own observations through on_observation after initialization and every
step, including the terminal step. run_game.py provides the human CLI session.
Future policies may retain these views or derive context from them. To
initialize such a policy partway through a game, reset it and deliver its
get_history_observation(None, player_id) results once, oldest first. Never pass
full SimulatorState records or another player's private observations.

The initial decision and encoder interfaces accept the current observation.
History inputs and recurrent model state can be added when a policy needs them;
the default hooks do not accumulate observations. Training loops, optimizers,
replay buffers, and reward calculations belong outside Player."""

from .base import ActionMask, Player, PlayerKind
from .human import HumanPlayer
from .neural import FlatObservationEncoder, QValueModel, NeuralPlayer, ObservationEncoder, SelectionMode
from .rule_based import (
    PlayDecision, RandomPlayer, RuleBasedPlayer, StructuredRuleBasedPlayer,
    RankingPlayer, WeightedRankingPlayer, RankingWeights, ScoredAction,
    RANKING_CONFIGS, create_ranking_player,
    ThresholdPlayer, ThresholdParameters, THRESHOLD_CONFIGS, create_threshold_player,
)


__all__ = [
    "ActionMask",
    "Player",
    "PlayerKind",
    "HumanPlayer",
    "NeuralPlayer",
    "FlatObservationEncoder",
    "RuleBasedPlayer",
    "RandomPlayer",
    "RankingPlayer",
    "WeightedRankingPlayer",
    "RankingWeights",
    "ScoredAction",
    "RANKING_CONFIGS",
    "create_ranking_player",
    "StructuredRuleBasedPlayer",
    "ThresholdPlayer",
    "ThresholdParameters",
    "THRESHOLD_CONFIGS",
    "create_threshold_player",
    "PlayDecision",
    "QValueModel",
    "ObservationEncoder",
    "SelectionMode",
]
