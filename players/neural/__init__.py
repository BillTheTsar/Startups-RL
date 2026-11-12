"""Q-value controller and observation encoder, usable without importing PyTorch.

Import QNetwork explicitly from players.neural.q_network when using PyTorch.
"""

from .encoder import FlatObservationEncoder
from .base import QValueModel, NeuralPlayer, ObservationEncoder, SelectionMode

__all__ = [
    "FlatObservationEncoder", "QValueModel", "NeuralPlayer",
    "ObservationEncoder", "SelectionMode",
]
