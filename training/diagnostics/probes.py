"""Fixed-observation diagnostics; no optimizer, simulator or random draws.

These measurements describe a supplied sample, not every reachable position.
Neuron inactivity means a ReLU output was zero throughout this sample; it does
not prove that the neuron can never activate. The model and its gradients are
left untouched. This module is separate from normal gameplay and training.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def distribution(values, *, return_bounds=False) -> dict:
    """Describe finite entries and explicitly count any nonfinite entries."""
    values = np.asarray(values, dtype=np.float64).ravel()
    finite = values[np.isfinite(values)]
    result = {"count": int(values.size), "nonfinite_count": int(values.size - finite.size)}
    if finite.size:
        result.update(zip(("min", "p01", "median", "p95", "p99", "max"),
                          map(float, np.quantile(finite, (0, .01, .5, .95, .99, 1)))))
        result.update(mean=float(finite.mean()), std=float(finite.std()))
        if return_bounds:
            result.update(below_minus_1_5_fraction=float(np.mean(finite < -1.5)),
                          above_1_5_fraction=float(np.mean(finite > 1.5)),
                          outside_return_range_fraction=float(np.mean(np.abs(finite) > 1.5)))
    return result


def load_probe_bank(path: str | Path) -> dict[str, np.ndarray]:
    """Load the fixed, pickle-free diagnostic observation bank into owned arrays."""
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key].copy() for key in saved.files}
    _validate_inputs(result["states"], result["masks"])
    if "strata" in result and len(result["strata"]) != len(result["states"]):
        raise ValueError("probe stratum labels must match the states")
    return result


def _validate_inputs(states, masks):
    states, masks = np.asarray(states), np.asarray(masks)
    if states.ndim != 2 or states.shape[1] != 332 or not len(states):
        raise ValueError("probe states must have nonempty shape [n,332]")
    if not np.issubdtype(states.dtype, np.floating) or not np.isfinite(states).all():
        raise ValueError("probe states must be finite floating-point values")
    if masks.dtype != np.bool_ or masks.shape != (len(states), 41) or not masks.any(axis=1).all():
        raise ValueError("probe masks must contain legal choices with shape [n,41]")
    return states, masks


def _summary(q_values, masks, pre_values, activations):
    max_legal = np.where(masks, q_values, -np.inf).max(axis=1)
    layers = []
    for index, (pre, activation) in enumerate(zip(pre_values, activations), start=1):
        positive = activation > 0
        active = positive.any(axis=0)
        zero_rows = (activation == 0).all(axis=1)
        layers.append({
            "layer": index, "width": activation.shape[1],
            "ever_active_units": int(active.sum()),
            "never_active_units": int((~active).sum()),
            "never_active_unit_ids": np.flatnonzero(~active).tolist(),
            "positive_fraction_by_unit": positive.mean(axis=0).tolist(),
            "all_zero_rows": int(zero_rows.sum()),
            "all_zero_fraction": float(zero_rows.mean()),
            "activation_fraction": float(positive.mean()),
            "positive_units_per_state": distribution(positive.sum(axis=1)),
            "pre_relu": distribution(pre), "activations": distribution(activation),
            "positive_activations": distribution(activation[positive]),
        })
    zero_last = (activations[-1] == 0).all(axis=1)
    return {
        "n_states": int(len(q_values)),
        "legal_q": distribution(q_values[masks], return_bounds=True),
        "max_legal_q": distribution(max_legal, return_bounds=True),
        "hidden_layers": layers,
        "last_layer_zero_vs_q": {
            "zero_last_layer_max_q": distribution(max_legal[zero_last], return_bounds=True),
            "nonzero_last_layer_max_q": distribution(max_legal[~zero_last], return_bounds=True),
            "above_1_5_and_zero_count": int(np.sum((max_legal > 1.5) & zero_last)),
            "above_10_and_zero_count": int(np.sum((max_legal > 10) & zero_last)),
        },
    }


@torch.inference_mode()
def measure_network(network, states, masks, *, batch_size=512, strata=None) -> dict:
    """Measure legal values and all hidden activations on identical observations.

    Accept either versioned QNetwork, including LayerNorm before ReLU in v2.
    Pre-ReLU measurements for v2 are after normalization. Preserve its train/eval flag,
    gradients, parameters and all random streams. Optional stratum labels add
    measurements for subsets without another network forward pass.
    """
    states, masks = _validate_inputs(states, masks)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if strata is not None:
        strata = np.asarray(strata)
        if strata.ndim != 1 or len(strata) != len(states):
            raise ValueError("strata must label each state")
    modules = list(network.layers)
    network.validate_architecture(allow_forward_hooks=True)
    parameter = next(network.parameters())
    all_q, all_pre, all_hidden = [], [[], [], []], [[], [], []]
    for start in range(0, len(states), batch_size):
        values = torch.as_tensor(states[start:start + batch_size], dtype=parameter.dtype,
                                 device=parameter.device)
        index = 0
        for module in modules[:-1]:
            if isinstance(module, nn.ReLU):
                all_pre[index].append(values.float().cpu().numpy())
                values = module(values)
                all_hidden[index].append(values.float().cpu().numpy())
                index += 1
            else:
                values = module(values)
        all_q.append(modules[-1](values).float().cpu().numpy())
    q_values = np.concatenate(all_q)
    pre_values = [np.concatenate(rows) for rows in all_pre]
    activations = [np.concatenate(rows) for rows in all_hidden]
    result = _summary(q_values, masks, pre_values, activations)
    result["output_bias"] = modules[-1].bias.detach().float().cpu().tolist()
    max_legal = np.where(masks, q_values, -np.inf).max(axis=1)
    top = np.argsort(max_legal)[-min(10, len(states)):][::-1]
    result["largest_q_examples"] = [{
        "probe_row": int(row), "q": float(max_legal[row]),
        "action_id": int(np.argmax(np.where(masks[row], q_values[row], -np.inf))),
        "positive_last_units": int(np.sum(activations[-1][row] > 0)),
        "last_activation_max": float(activations[-1][row].max()),
        "max_abs_input": float(np.abs(states[row]).max()),
    } for row in top]
    if strata is not None:
        result["by_stratum"] = {}
        for label in np.unique(strata):
            selected = strata == label
            result["by_stratum"][str(label)] = _summary(
                q_values[selected], masks[selected], [x[selected] for x in pre_values],
                [x[selected] for x in activations],
            )
    return result
