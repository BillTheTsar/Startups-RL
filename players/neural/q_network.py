"""Versioned feed-forward Startups Q-networks and their inference adapter.

Import this module only when PyTorch is needed. The simulator, observation
encoder and framework-independent player interface do not depend on it.
The network sees one 332-value observation and returns 41 raw Q-values in
ActionCodec order. Legal-action masks belong to selection and target building,
not to this network's inputs or its output activation.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from action_codec import ActionCodec

from .encoder import FlatObservationEncoder


class QNetwork(nn.Module):
    """Map 332 inputs through 256, 256 and 128 units to 41 action values.

    Version 1 places ReLU after each hidden linear layer. Version 2 inserts
    LayerNorm before each of those ReLUs. Both use the same widths and a final
    linear layer, allowing negative
    values and values above one; no softmax or probability interpretation is
    applied. Initialize float32 parameters on CPU, then use ordinary Module.to
    calls to select a device/dtype. Initialization uses PyTorch's current random
    generator, so the training setup owns seeding.

    forward retains autograd for optimization. predict adapts one encoded
    vector for a NeuralPlayer, without gradients or changing shared model state.
    This module owns no optimizer, target network, action sampler or replay data.
    """

    # Keep the historical no-argument constructor and class constant unchanged.
    # A particular model's version is available through architecture_version.
    ARCHITECTURE_VERSION = 1
    SUPPORTED_ARCHITECTURE_VERSIONS = (1, 2)
    LAYER_NORM_EPS = 1e-5
    INPUT_SIZE = FlatObservationEncoder.SIZE
    HIDDEN_SIZES = (256, 256, 128)
    OUTPUT_SIZE = ActionCodec.ACTION_COUNT

    def __init__(self, architecture_version: int = 1) -> None:
        """Construct v1 or v2, retaining the exact v1 parameter names and order.

        LayerNorm uses learned scale and offset, with no running statistics.
        It normalizes hidden features, not the 41 output Q-values; it does not
        impose a bound on those predictions.
        """
        self._validate_architecture_version(architecture_version)
        super().__init__()
        self._architecture_version = architecture_version
        widths = (self.INPUT_SIZE, *self.HIDDEN_SIZES, self.OUTPUT_SIZE)
        layers: list[nn.Module] = []
        for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(inputs, outputs, device="cpu", dtype=torch.float32))
            if index < len(self.HIDDEN_SIZES):
                if architecture_version == 2:
                    layers.append(nn.LayerNorm(outputs, eps=self.LAYER_NORM_EPS,
                                               elementwise_affine=True,
                                               device="cpu", dtype=torch.float32))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    @property
    def architecture_version(self) -> int:
        """Identify the layout used by this network, independently of its dtype."""
        return self._architecture_version

    @classmethod
    def _validate_architecture_version(cls, architecture_version: int) -> None:
        if type(architecture_version) is not int or architecture_version not in cls.SUPPORTED_ARCHITECTURE_VERSIONS:
            raise ValueError("architecture_version must be 1 or 2")

    @classmethod
    def parameter_shapes(cls, architecture_version: int = 1) -> dict[str, tuple[int, ...]]:
        """Return expected state-dict keys and shapes without allocating a model.

        Checkpoint and catalogue validation can use this without consuming RNG
        state. Every call returns a new dictionary.
        """
        cls._validate_architecture_version(architecture_version)
        widths = (cls.INPUT_SIZE, *cls.HIDDEN_SIZES, cls.OUTPUT_SIZE)
        shapes = {}
        layer_index = 0
        for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
            shapes[f"layers.{layer_index}.weight"] = (outputs, inputs)
            shapes[f"layers.{layer_index}.bias"] = (outputs,)
            layer_index += 1
            if index < len(cls.HIDDEN_SIZES):
                if architecture_version == 2:
                    shapes[f"layers.{layer_index}.weight"] = (outputs,)
                    shapes[f"layers.{layer_index}.bias"] = (outputs,)
                    layer_index += 1
                layer_index += 1  # ReLU has no state-dict entries.
        return shapes

    @classmethod
    def architecture_metadata(cls, architecture_version: int = 1) -> dict[str, object]:
        """Return the exact inference descriptor for a supported layout.

        The v1 descriptor is unchanged so existing permanent policy identities
        remain stable. V2 records the normalization settings as part of identity.
        """
        cls._validate_architecture_version(architecture_version)
        metadata = {"name": "startups.q_network", "version": architecture_version,
                    "layers": [cls.INPUT_SIZE, *cls.HIDDEN_SIZES, cls.OUTPUT_SIZE],
                    "hidden_activation": "relu", "output": "q_values"}
        if architecture_version == 2:
            metadata["hidden_normalization"] = {
                "name": "layer_norm", "eps": cls.LAYER_NORM_EPS,
                "elementwise_affine": True, "position": "before_relu",
            }
        return metadata

    def validate_architecture(self, *, allow_forward_hooks: bool = False) -> None:
        """Reject runtime changes that cannot be represented by the descriptor.

        Saving weights must not silently discard changed normalization settings,
        activation behavior, custom modules or forward hooks. Diagnostic readers
        may explicitly allow observation hooks; storage always uses the strict
        default so saved models never discard altered forward behavior. Tensor integrity
        and dtype checks remain the responsibility of checkpoint/artifact I/O.
        """
        self._validate_architecture_version(self.architecture_version)
        if type(self) is not QNetwork or type(self.layers) is not nn.Sequential:
            raise ValueError("network must use an unmodified QNetwork architecture")
        if any(getattr(self, name) != getattr(QNetwork, name) for name in
               ("INPUT_SIZE", "HIDDEN_SIZES", "OUTPUT_SIZE", "LAYER_NORM_EPS")):
            raise ValueError("network settings do not match its architecture")
        widths = (self.INPUT_SIZE, *self.HIDDEN_SIZES, self.OUTPUT_SIZE)
        modules = list(self.layers)
        stride = 2 if self.architecture_version == 1 else 3
        if len(modules) != stride * len(self.HIDDEN_SIZES) + 1:
            raise ValueError("network layers do not match its architecture_version")
        for index, (inputs, outputs) in enumerate(zip(widths, widths[1:])):
            linear = modules[stride * index]
            if type(linear) is not nn.Linear or linear.in_features != inputs or (
                linear.out_features != outputs or linear.bias is None
            ):
                raise ValueError("network linear layer does not match its architecture")
            if index < len(self.HIDDEN_SIZES):
                relu = modules[stride * index + stride - 1]
                if type(relu) is not nn.ReLU or relu.inplace:
                    raise ValueError("network ReLU does not match its architecture")
                if self.architecture_version == 2:
                    norm = modules[stride * index + 1]
                    if type(norm) is not nn.LayerNorm or norm.normalized_shape != (outputs,) or (
                        norm.eps != self.LAYER_NORM_EPS or not norm.elementwise_affine
                        or norm.weight is None or norm.bias is None
                    ):
                        raise ValueError("network LayerNorm does not match its architecture")
        for module in self.modules():
            if ((module._forward_hooks or module._forward_pre_hooks) and not allow_forward_hooks) or "forward" in vars(module):
                raise ValueError("network forward behavior does not match its architecture")

    def forward(self, states: Tensor) -> Tensor:
        """Return Q-values for [332] or [batch, 332] floating-point inputs.

        Preserve the leading batch dimension and autograd graph. Require the
        input device and dtype to match this model; training code owns batch
        conversion. An empty batch is allowed. Only structural validation is
        performed here, avoiding repeated GPU synchronization during training.
        """
        if not isinstance(states, Tensor):
            raise TypeError("states must be a torch.Tensor")
        if states.ndim not in (1, 2) or states.shape[-1] != self.INPUT_SIZE:
            raise ValueError("states must have shape [332] or [batch, 332]")
        if not states.is_floating_point():
            raise ValueError("states must have a floating-point dtype")
        parameter = next(self.parameters())
        if states.dtype != parameter.dtype or states.device != parameter.device:
            raise ValueError("states dtype and device must match the Q-network")
        return self.layers(states)

    def predict(self, encoded: Sequence[float] | Tensor) -> tuple[float, ...]:
        """Evaluate one encoded observation as 41 immutable Python float values.

        Accept the encoder's array('f'), a numerical sequence, or a 1-D tensor.
        Convert to this network's current device/dtype and run under inference
        mode. Reject malformed or nonfinite inputs. Do not change train/eval
        flags, parameters, existing gradients, or the input collection. The
        architecture has no dropout or batch normalization, so inference does
        not require temporarily toggling a shared network's mode.

        Pass network.predict as the model callable supplied to NeuralPlayer.
        Use forward directly when computing a differentiable training loss.
        """
        parameter = next(self.parameters())
        with torch.inference_mode():
            try:
                states = torch.as_tensor(encoded, device=parameter.device)
            except (TypeError, ValueError, RuntimeError, OverflowError) as error:
                raise ValueError("encoded observation must be a numerical vector of length 332") from error
            if states.ndim != 1 or states.shape[0] != self.INPUT_SIZE:
                raise ValueError("predict requires one encoded observation of length 332")
            if states.is_complex() or states.dtype == torch.bool:
                raise ValueError("encoded observation must contain real numbers")
            states = states.to(dtype=parameter.dtype)
            if not bool(torch.isfinite(states).all()):
                raise ValueError("encoded observation must contain finite representable values")
            values = self(states)
            return tuple(values.detach().cpu().tolist())
