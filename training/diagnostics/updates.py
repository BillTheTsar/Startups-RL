"""Observe a DQN or Double DQN update without changing its learning calculation.

The production learner still performs the forward pass, loss, backward pass,
AdamW step and optional target update. Scoped hooks read those operations. This
module owns no replay sampling, random stream, training schedule or file output.
"""

from collections.abc import Callable
from math import isfinite, sqrt

import numpy as np
import torch
from torch import nn

from training.batches import ArrayBatch
from training.dqn import DQNLearner, UpdateResult


BeforeStep = Callable[[DQNLearner, ArrayBatch, dict], None]


def _number(value) -> float | None:
    """Represent nonfinite diagnostic scalars explicitly rather than invalid JSON."""
    value = float(value)
    return value if isfinite(value) else None


def _distribution(values) -> dict:
    """Summarize a small detached vector, retaining nonfinite counts separately."""
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    result = {"count": int(values.size), "nonfinite_count": int(values.size - finite.size)}
    if not finite.size:
        result.update({name: None for name in (
            "mean", "min", "p50", "p95", "p99", "max", "abs_max", "outside_return_fraction",
        )})
        return result
    q50, q95, q99 = np.quantile(finite, [0.5, 0.95, 0.99])
    result.update(mean=float(finite.mean()), min=float(finite.min()), p50=float(q50),
                  p95=float(q95), p99=float(q99), max=float(finite.max()),
                  abs_max=float(np.abs(finite).max()),
                  outside_return_fraction=float(np.mean(np.abs(finite) > 1.5)))
    return result


def _error_group(errors: np.ndarray, selected: np.ndarray) -> dict:
    values = errors[selected].astype(np.float64)
    return {"count": int(values.size),
            "td_mse": _number(np.mean(values * values)) if values.size else None,
            "td_abs_mean": _number(np.mean(np.abs(values))) if values.size else None,
            "td_abs_max": _number(np.max(np.abs(values))) if values.size else None}


def _activation_stats(preactivations: dict[str, torch.Tensor]) -> dict:
    """Copy the small activation batch once, avoiding many tiny CUDA reductions."""
    if not preactivations:
        return {}
    packed = torch.cat(tuple(preactivations.values()), dim=1).float().cpu().numpy()
    result = {}
    offset = 0
    for name, tensor in preactivations.items():
        width = tensor.shape[1]
        pre = packed[:, offset:offset + width]
        zero = pre <= 0
        result[name] = {
            "neurons": width, "active_neurons": int((pre > 0).any(axis=0).sum()),
            "zero_fraction": float(zero.mean()), "all_zero_row_fraction": float(zero.all(axis=1).mean()),
            "pre_min": _number(pre.min()), "pre_mean": _number(pre.mean()), "pre_max": _number(pre.max()),
            "near_zero_fraction": float((np.abs(pre) < 0.01).mean()),
            "nonfinite_count": int((~np.isfinite(pre)).sum()),
        }
        offset += width
    return result


class UpdateObserver:
    """Return one ordinary learner update together with measurements of that step.

    before_step, when supplied, runs after backpropagation and before AdamW. It
    receives (learner, batch, summary), allowing the caller to save the exact
    pre-update model/optimizer and batch when an incident merits inspection.
    It must only read those objects and must not consume training RNG streams.
    An exception from the callback prevents that optimizer step and propagates.

    The observer itself makes no extra model forward/backward pass and consumes
    no random numbers. It supports ArrayBatch, the production training path.
    The layer names describe the current Linear/ReLU QNetwork architecture.
    Activation summaries describe this update's batch; a separate fixed set of
    observations is required to compare representations across different steps.
    Double DQN's online next-state pass runs without gradients and is excluded
    from current-state predictions and activation measurements.
    """

    def __init__(self, before_step: BeforeStep | None = None) -> None:
        if before_step is not None and not callable(before_step):
            raise ValueError("before_step must be callable or None")
        self.before_step = before_step

    def update(self, learner: DQNLearner, batch: ArrayBatch) -> tuple[UpdateResult, dict]:
        if not isinstance(learner, DQNLearner) or not isinstance(batch, ArrayBatch):
            raise TypeError("UpdateObserver requires a DQNLearner and ArrayBatch")
        linears = [(index, layer) for index, layer in enumerate(learner.online.layers)
                   if isinstance(layer, nn.Linear)]
        groups = {
            ("output" if order == len(linears) - 1 else f"hidden_{order + 1}"):
                tuple(layer.parameters())
            for order, (_, layer) in enumerate(linears)
        }
        before = {name: torch.cat([parameter.detach().flatten() for parameter in parameters])
                  for name, parameters in groups.items()}
        captured, preactivations, handles = {}, {}, []
        step_before = learner.optimizer_steps
        summary = {"step_before": step_before, "step": step_before + 1,
                   "batch_size": len(batch), "terminal_count": int(batch.done.sum()),
                   "target_updated": (step_before + 1) % learner.config.target_update_interval == 0}

        def capture_preactivation(name):
            def capture(_module, _inputs, output):
                if torch.is_grad_enabled():
                    preactivations[name] = output.detach()
            return capture

        def capture_output(_module, _inputs, output):
            if torch.is_grad_enabled():
                captured["predictions"] = output.detach()

        original_targets = learner._targets
        had_targets_override = "_targets" in learner.__dict__
        prior_targets_override = learner.__dict__.get("_targets")

        def capture_targets(current_batch):
            targets = original_targets(current_batch)
            captured["targets"] = targets.detach()
            return targets

        def observe_before_step(_optimizer, _args, _kwargs):
            predictions = captured["predictions"].float().cpu().numpy()
            targets = captured["targets"].float().cpu().numpy()
            selected = predictions[np.arange(len(batch)), batch.actions]
            max_legal = np.where(batch.masks, predictions, -np.inf).max(axis=1)
            errors = selected - targets
            summary.update(
                selected_q=_distribution(selected), max_legal_q=_distribution(max_legal),
                legal_q=_distribution(predictions[batch.masks]), targets=_distribution(targets),
                td_error=_distribution(errors), abs_td_error=_distribution(np.abs(errors)),
                terminal=_error_group(errors, batch.done),
                nonterminal=_error_group(errors, ~batch.done),
                loss=_number(np.mean(errors.astype(np.float64) ** 2)),
                batch_activations=_activation_stats(preactivations),
            )
            reductions = []
            for name, parameters in groups.items():
                gradients = [parameter.grad.detach().flatten() for parameter in parameters if parameter.grad is not None]
                zero = parameters[0].detach().new_zeros(())
                gradient = torch.cat(gradients) if gradients else zero.reshape(1)
                reductions.extend((
                    torch.linalg.vector_norm(gradient, dtype=torch.float64), gradient.abs().max(),
                    torch.linalg.vector_norm(before[name], dtype=torch.float64),
                    (~torch.isfinite(gradient)).sum().float(),
                ))
            values = torch.stack(reductions).detach().cpu().tolist()
            summary["layers"] = {}
            grad_squared_total, parameter_squared_total, gradient_max = 0.0, 0.0, 0.0
            nonfinite_total = 0
            for offset, name in enumerate(groups):
                grad_norm, grad_max, parameter_norm, nonfinite = values[offset * 4:offset * 4 + 4]
                summary["layers"][name] = {
                    "gradient_l2": _number(grad_norm), "gradient_abs_max": _number(grad_max),
                    "parameter_l2_before": _number(parameter_norm),
                    "gradient_nonfinite_count": int(nonfinite),
                }
                grad_squared_total += grad_norm ** 2
                parameter_squared_total += parameter_norm ** 2
                gradient_max = max(gradient_max, grad_max)
                nonfinite_total += int(nonfinite)
            summary.update(gradient_l2=_number(sqrt(grad_squared_total)),
                           gradient_abs_max=_number(gradient_max),
                           gradient_nonfinite_count=nonfinite_total,
                           parameter_l2_before=_number(sqrt(parameter_squared_total)))
            if self.before_step is not None:
                self.before_step(learner, batch, summary)

        try:
            for order, (_, layer) in enumerate(linears[:-1], 1):
                handles.append(layer.register_forward_hook(capture_preactivation(f"hidden_{order}")))
            handles.append(learner.online.register_forward_hook(capture_output))
            handles.append(learner.optimizer.register_step_pre_hook(observe_before_step))
            learner._targets = capture_targets
            result = learner.update(batch)
        finally:
            if had_targets_override:
                learner._targets = prior_targets_override
            else:
                learner.__dict__.pop("_targets", None)
            for handle in handles:
                handle.remove()

        # AdamW has completed, including momentum and decoupled weight decay.
        # Target synchronization changes only target parameters, not these deltas.
        deltas = torch.stack([
            torch.linalg.vector_norm(torch.cat([parameter.detach().flatten() for parameter in parameters]) - before[name], dtype=torch.float64)
            for name, parameters in groups.items()
        ]).detach().cpu().tolist()
        for name, delta_norm in zip(groups, deltas):
            delta = _number(delta_norm)
            baseline = summary["layers"][name]["parameter_l2_before"]
            summary["layers"][name].update(
                update_l2=delta,
                relative_update_l2=delta / baseline if delta is not None and baseline else None,
            )
        total_delta = _number(sqrt(sum(value ** 2 for value in deltas)))
        baseline = summary["parameter_l2_before"]
        summary.update(step=result.step, loss=result.loss, target_updated=result.target_updated,
                       update_l2=total_delta,
                       relative_update_l2=total_delta / baseline if total_delta is not None and baseline else None)
        return result, summary
