"""One reusable DQN optimizer update, independent of collection and persistence.

Ordinary DQN uses the target network's maximum legal next-action value.
Double DQN lets the online network select a legal next action and the target
network evaluate it. Terminal transitions never enter next-state inference.
Replay composition, epsilon schedules, games, evaluation, and checkpoint storage
belong to callers of this module.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from players.neural.q_network import QNetwork

from .batches import ArrayBatch
from .config import TrainingConfig
from .transitions import Transition


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """A completed optimizer update and its pre-update mean squared TD error."""

    step: int
    loss: float
    target_updated: bool


def _resolve_device(requested: str) -> torch.device:
    """Resolve auto or an explicit validated CPU/CUDA choice before allocation."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    index = int(requested.split(":", 1)[1]) if ":" in requested else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is not available")
    return torch.device("cuda", index)


class DQNLearner:
    """Own online/target networks, AdamW, and the global optimizer-step counter.

    A supplied online QNetwork becomes this learner's trainable network: it is
    moved to the configured device and put in training mode. Its weights are
    preserved. It must match config.architecture_version. Otherwise create the
    selected 332 -> 256 -> 256 -> 128 -> 41 model, with LayerNorm in version 2.
    The independent target starts with identical weights, in evaluation mode
    with gradients disabled. Only the online parameters belong to AdamW.

    Seeded construction restores the caller's CPU RNG state and does not seed
    CUDA generators. With no seed, ordinary PyTorch initialization consumes its
    CPU RNG. No model initialization is performed when online is supplied.
    """

    def __init__(
        self, config: TrainingConfig | None = None, *, online: QNetwork | None = None,
    ) -> None:
        self.config = TrainingConfig() if config is None else config
        if not isinstance(self.config, TrainingConfig):
            raise ValueError("config must be a TrainingConfig")
        if online is not None and not isinstance(online, QNetwork):
            raise ValueError("online must be a QNetwork or None")
        if online is not None and online.architecture_version != self.config.architecture_version:
            raise ValueError("online architecture_version must match the training configuration")
        self.device = _resolve_device(self.config.device)
        if online is None:
            if self.config.seed is None:
                online = QNetwork(architecture_version=self.config.architecture_version)
            else:
                # QNetwork initializes on CPU. Seeding that generator directly
                # avoids torch.manual_seed's additional CUDA-generator changes.
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(self.config.seed % (2**64))
                    online = QNetwork(architecture_version=self.config.architecture_version)
        self.online = online.to(self.device).train().requires_grad_(True)
        self.target = deepcopy(self.online).eval().requires_grad_(False)
        # A supplied network may already have gradients; a target never needs
        # copied gradient storage, even before the first optimizer update.
        self.target.zero_grad(set_to_none=True)
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer_steps = 0

    @staticmethod
    def _batch(batch: Sequence[Transition] | ArrayBatch) -> tuple[Transition, ...] | ArrayBatch:
        """Validate array contents once, or require existing Transition objects.

        Arrays are owned by their batch but may be edited by its caller. A
        vectorized check catches invalid edits without rebuilding individual
        transitions or visiting every state value in Python.
        """
        if isinstance(batch, ArrayBatch):
            batch.validate()
            return batch
        try:
            entries = tuple(batch)
        except TypeError as error:
            raise ValueError("batch must be a nonempty sequence of Transition objects") from error
        if not entries or any(not isinstance(entry, Transition) for entry in entries):
            raise ValueError("batch must be a nonempty sequence of Transition objects")
        return entries

    @torch.no_grad()
    def _targets(self, batch: tuple[Transition, ...] | ArrayBatch, *, unclipped: bool = False) -> Tensor:
        """Compute detached targets and optionally constrain finite return estimates.

        Clip only after reward and discount, never before next-action selection.
        Preserve NaN/Infinity so update still rejects an invalid target instead
        of turning it into a plausible finite training label.
        """
        targets = self._raw_targets(batch)
        if self.config.clip_targets and not unclipped:
            targets = torch.where(torch.isfinite(targets), targets.clamp(-1.5, 1.5), targets)
        return targets

    @torch.no_grad()
    def _raw_targets(self, batch: tuple[Transition, ...] | ArrayBatch) -> Tensor:
        """Compute targets before clipping, evaluating only nonterminal next states."""
        if isinstance(batch, ArrayBatch):
            return self._array_targets(batch)
        dtype = next(self.online.parameters()).dtype
        targets = torch.tensor([entry.reward for entry in batch], dtype=dtype, device=self.device)
        pending = [index for index, entry in enumerate(batch) if not entry.done]
        if pending and self.config.gamma != 0:
            states = torch.tensor([batch[index].next_state for index in pending], dtype=dtype, device=self.device)
            masks = torch.tensor([batch[index].next_mask for index in pending], dtype=torch.bool, device=self.device)
            targets[pending] += self.config.gamma * self._next_state_values(states, masks)
        return targets

    @torch.no_grad()
    def _array_targets(self, batch: ArrayBatch) -> Tensor:
        """Use contiguous arrays without reconstructing Python state tuples.

        The caller has validated this batch. Clone the reward tensor because
        a CPU float64 learner could otherwise share the caller's reward array.
        Terminal rows are excluded before moving next states to the device.
        """
        dtype = next(self.online.parameters()).dtype
        targets = torch.from_numpy(batch.rewards).to(device=self.device, dtype=dtype).clone()
        pending = np.flatnonzero(~batch.done).tolist()
        if pending and self.config.gamma != 0:
            states = torch.from_numpy(batch.next_states[pending]).to(device=self.device, dtype=dtype)
            masks = torch.from_numpy(batch.next_masks[pending]).to(device=self.device)
            targets[pending] += self.config.gamma * self._next_state_values(states, masks)
        return targets

    @torch.no_grad()
    def _next_state_values(self, states: Tensor, masks: Tensor) -> Tensor:
        """Evaluate nonterminal next states according to the configured algorithm.

        Every row has at least one legal action, checked during batch validation.
        DQN selects and evaluates with the target network. Double DQN selects
        with masked online values, then gathers the target value for that action.
        Ties select the lowest legal action ID, as in greedy NeuralPlayer play.

        Illegal nonfinite predictions are ignored. A nonfinite legal online
        value makes Double DQN's selection unreliable: propagate a NaN target
        so update's existing finite-loss check rejects the batch before changing
        gradients or optimizer state. Neither next-state pass records gradients.
        """
        if self.config.algorithm == "double_dqn":
            online_values = self.online(states).masked_fill(~masks, -torch.inf)
            next_actions = online_values.argmax(dim=1, keepdim=True)
            values = self.target(states).gather(1, next_actions).squeeze(1)
            invalid_selection = (~torch.isfinite(online_values) & masks).any(dim=1)
            return values.masked_fill(invalid_selection, torch.nan)
        return self.target(states).masked_fill(~masks, -torch.inf).max(dim=1).values

    def compute_targets(self, batch: Sequence[Transition] | ArrayBatch, *, unclipped: bool = False) -> Tensor:
        """Return detached targets for the configured DQN variant for inspection.

        DQN: r + gamma * max_legal Q_target(s', a'). Double DQN: the online
        network selects the best legal next action; the target network supplies
        that action's value. Only next observations participate in this selection.

        With clip_targets enabled, finite targets are clamped to [-1.5, 1.5].
        Pass unclipped=True to inspect the estimates before that operation.
        Network outputs are never clipped. Nonfinite targets remain nonfinite.

        Use only r when done=True, including terminal rewards of zero. Never
        evaluate terminal next states or maximize their empty masks. Gamma zero
        also skips both networks' next-state inference. The returned vector is
        fresh, on the learner's device and in the online network's dtype.
        This does not change weights, gradients, optimizer state or step counts.
        Accept existing Transition sequences or compact ArrayBatch inputs.
        Array batches are checked once with vectorized operations; Transition
        construction already validates sequence entries. Neither path changes
        the supplied batch.
        """
        if type(unclipped) is not bool:
            raise ValueError("unclipped must be a boolean")
        return self._targets(self._batch(batch), unclipped=unclipped)

    @torch.no_grad()
    def soft_update_target(self) -> None:
        """Apply target <- (1 - tau) * target + tau * online without a step.

        update calls this only at configured optimizer-step intervals. The
        supported architectures have no running buffers to synchronize.
        V2 LayerNorm's learned scale and offset are blended with other parameters.
        Target parameters stay frozen; there is no reverse gradient connection.
        """
        for target, online in zip(self.target.parameters(), self.online.parameters(), strict=True):
            target.lerp_(online, self.config.target_tau)

    def update(self, batch: Sequence[Transition] | ArrayBatch) -> UpdateResult:
        """Optimize the mean squared TD error, then advance the global counter.

        Gather only Q_online(s, action_id) for each recorded action. DQN targets
        use the target network's legal maximum. Double DQN targets use its value
        for the online network's best legal next action. Construct targets before
        the current-state forward pass; only that current-state pass is trained.
        All sampled transitions have equal weight regardless of replay source.
        Optional target clipping constrains the finite training labels, not the
        current predictions. MSE remains the only loss; there is no extra penalty.
        Any nonempty batch size is supported; replay owns the partition mix.
        ArrayBatch inputs avoid Python tuple reconstruction and are converted
        directly from their owned arrays. Existing Transition sequences remain
        supported. The caller must not edit array inputs during this call.

        Reject malformed entries or a nonfinite loss before touching optimizer
        state or clearing existing gradients. After AdamW succeeds, increment
        optimizer_steps and softly update the target if this global step is a
        multiple of target_update_interval. No game boundary resets the count.
        """
        batch = self._batch(batch)
        dtype = next(self.online.parameters()).dtype
        if isinstance(batch, ArrayBatch):
            states = torch.from_numpy(batch.states).to(device=self.device, dtype=dtype)
            actions = torch.from_numpy(batch.actions).to(device=self.device, dtype=torch.long)
        else:
            states = torch.tensor([entry.state for entry in batch], dtype=dtype, device=self.device)
            actions = torch.tensor([entry.action_id for entry in batch], dtype=torch.long, device=self.device)
        targets = self._targets(batch)
        selected_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(selected_values, targets, reduction="mean")
        loss_value = float(loss.detach().cpu())
        if not isfinite(loss_value):
            raise ValueError("DQN loss must be finite; optimizer update was not applied")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.optimizer_steps += 1
        target_updated = self.optimizer_steps % self.config.target_update_interval == 0
        if target_updated:
            self.soft_update_target()
        return UpdateResult(self.optimizer_steps, loss_value, target_updated)
