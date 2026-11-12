"""Replay saved diagnostic updates offline, without collecting any games.

Each incident contains the online/target networks, AdamW state and exact batch
immediately before one update. Restore those objects, measure the fixed probe
bank, perform that one unchanged production update, then measure again. The
archived observer source verifies that all measurements match the live trace.
Neither production training files nor the live diagnostic trace are changed.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from players.neural.q_network import QNetwork
from training.batches import ArrayBatch
from training.config import TrainingConfig
from training.dqn import DQNLearner
from training.runtime import configure_cpu_threads

from .probes import distribution, load_probe_bank, measure_network


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_observer(path):
    specification = importlib.util.spec_from_file_location("archived_diagnostic_observer", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.UpdateObserver


def _differences(expected, actual, path=""):
    result = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            result.append({"path": path, "expected_keys": sorted(expected),
                           "actual_keys": sorted(actual) if isinstance(actual, dict) else None})
        if isinstance(actual, dict):
            for key in expected.keys() & actual.keys():
                result.extend(_differences(expected[key], actual[key], f"{path}.{key}".lstrip(".")))
    elif expected != actual or type(expected) is not type(actual):
        item = {"path": path, "expected": expected, "actual": actual}
        if type(expected) in (float, int) and type(actual) in (float, int):
            item["absolute_difference"] = abs(expected - actual)
        result.append(item)
    return result


def _matching_fields(expected, actual):
    """Restrict a final summary to the fields already available before a step."""
    if isinstance(expected, dict):
        return {name: _matching_fields(value, actual[name])
                for name, value in expected.items()}
    return actual


def _read_live_records(path, requested_steps):
    """Read the append-only trace once, decoding only requested completed rows."""
    requested_steps = set(requested_steps)
    needles = {step: f'"step":{step},' for step in requested_steps}
    found = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                continue
            for step, needle in tuple(needles.items()):
                if needle in line:
                    record = json.loads(line)
                    if record.get("step") == step:
                        found[step] = record
                        del needles[step]
            if not needles:
                break
    return found


def _restore(payload, device):
    if payload.get("diagnostic_version") != 1 or payload.get("kind") != "diagnostic_before_update":
        raise ValueError("expected a saved pre-update diagnostic incident")
    config = replace(TrainingConfig.from_dict(payload["config"]), device=device)
    configure_cpu_threads(config.cpu_threads)
    online = QNetwork()
    online.load_state_dict(payload["online"], strict=True)
    learner = DQNLearner(config, online=online)
    learner.target.load_state_dict(payload["target"], strict=True)
    learner.optimizer.load_state_dict(payload["optimizer"])
    learner.optimizer_steps = payload["optimizer_steps"]
    batch = ArrayBatch(**{name: payload["batch"][name].detach().cpu().numpy()
                          for name in ArrayBatch.FIELD_NAMES})
    rng = payload["global_rng"]
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available():
        for index, state in enumerate(rng["torch_cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(state, index)
    return learner, batch


def _probe_snapshot(network, bank):
    """Measure once, retaining row identities needed for before/after comparisons."""
    zeros = [[], [], []]
    q_values = []
    handles = []

    def capture_zero(index):
        def capture(_module, _inputs, output):
            zeros[index].append((output.detach() == 0).all(dim=1).cpu().numpy())
        return capture

    def capture_q(_module, _inputs, output):
        q_values.append(output.detach().float().cpu().numpy())

    try:
        relus = [module for module in network.layers if isinstance(module, nn.ReLU)]
        for index, module in enumerate(relus):
            handles.append(module.register_forward_hook(capture_zero(index)))
        handles.append(network.layers[-1].register_forward_hook(capture_q))
        summary = measure_network(network, bank["states"], bank["masks"], strata=bank.get("strata"))
    finally:
        for handle in handles:
            handle.remove()
    return summary, [np.concatenate(rows) for rows in zeros], np.concatenate(q_values)


def _probe_changes(before, after, before_zero, after_zero, before_q, after_q, masks):
    layers = []
    for old, new, old_zero, new_zero in zip(before["hidden_layers"], after["hidden_layers"],
                                          before_zero, after_zero):
        before_active = set(range(old["width"])) - set(old["never_active_unit_ids"])
        after_active = set(range(new["width"])) - set(new["never_active_unit_ids"])
        layers.append({
            "layer": old["layer"], "active_units_before": sorted(before_active),
            "active_units_after": sorted(after_active),
            "units_that_stopped_activating": sorted(before_active - after_active),
            "units_that_started_activating": sorted(after_active - before_active),
            "all_zero_rows_before": int(old_zero.sum()), "all_zero_rows_after": int(new_zero.sum()),
            "newly_all_zero_probe_rows": np.flatnonzero(new_zero & ~old_zero).tolist(),
            "newly_nonzero_probe_rows": np.flatnonzero(old_zero & ~new_zero).tolist(),
        })
    before_max = np.where(masks, before_q, -np.inf).max(axis=1)
    after_max = np.where(masks, after_q, -np.inf).max(axis=1)
    before_choice = np.where(masks, before_q, -np.inf).argmax(axis=1)
    after_choice = np.where(masks, after_q, -np.inf).argmax(axis=1)
    return {
        "hidden_layers": layers,
        "max_legal_q_change": distribution(after_max - before_max),
        "all_legal_q_change": distribution((after_q - before_q)[masks]),
        "greedy_action_changed_count": int(np.sum(before_choice != after_choice)),
        "greedy_action_changed_fraction": float(np.mean(before_choice != after_choice)),
    }


def _state_context(state):
    """Decode a few familiar, viewer-relative fields for inspecting outliers."""
    companies = range(5, 11)
    return {
        "phase": ("take", "play", "finished")[int(np.argmax(state[:3]))],
        "deck_size": int(round(float(state[3]) * 28)),
        "hand_counts": {str(company): int(round(float(state[16 + index]) * company))
                        for index, company in enumerate(companies)},
        "player_coins_self_then_clockwise": [int(round(float(state[34 + 14 * index]) * 15))
                                            for index in range(4)],
        "market_size": sum(bool(state[108 + 8 * index] > 0.5) for index in range(28)),
        "max_abs_encoded_input": float(np.abs(state).max()),
    }


@torch.no_grad()
def _batch_values(learner, batch):
    parameter = next(learner.online.parameters())
    states = torch.from_numpy(batch.states).to(device=parameter.device, dtype=parameter.dtype)
    return learner.online(states).float().cpu().numpy(), learner.compute_targets(batch).float().cpu().numpy()


def analyse_incident(path, *, bank, observer_class, live_record=None, device="cuda"):
    path = Path(path)
    source_stat = path.stat()
    with torch.serialization.safe_globals([bytes]):
        payload = torch.load(path, weights_only=True, map_location="cpu")
    learner, batch = _restore(payload, device)
    before_summary, before_zero, before_probe_q = _probe_snapshot(learner.online, bank)
    before_q, targets = _batch_values(learner, batch)
    result, update_summary = observer_class().update(learner, batch)
    after_summary, after_zero, after_probe_q = _probe_snapshot(learner.online, bank)
    after_q, after_targets = _batch_values(learner, batch)
    rows = []
    for index in range(len(batch)):
        action = int(batch.actions[index])
        greedy = int(np.argmax(np.where(batch.masks[index], before_q[index], -np.inf)))
        after_greedy = int(np.argmax(np.where(batch.masks[index], after_q[index], -np.inf)))
        rows.append({
            "row": index, "partition": "demo" if index < learner.config.demo_batch_size else "non_demo",
            "recorded_action_id": action, "greedy_action_id_before": greedy,
            "greedy_action_id_after": after_greedy, "recorded_action_is_greedy": action == greedy,
            "selected_q_before": float(before_q[index, action]),
            "selected_q_after": float(after_q[index, action]),
            "max_legal_q_before": float(before_q[index, greedy]),
            "max_legal_q_after": float(after_q[index, after_greedy]),
            "same_greedy_action_q_after": float(after_q[index, greedy]),
            "target_before": float(targets[index]), "target_after": float(after_targets[index]),
            "selected_td_before": float(before_q[index, action] - targets[index]),
            "selected_td_after_against_same_target": float(after_q[index, action] - targets[index]),
            "reward": float(batch.rewards[index]), "done": bool(batch.done[index]),
            "legal_action_ids": np.flatnonzero(batch.masks[index]).tolist(),
            "state": _state_context(batch.states[index]),
        })
    top_q = sorted(rows, key=lambda row: abs(row["max_legal_q_before"]), reverse=True)[:10]
    top_td = sorted(rows, key=lambda row: abs(row["selected_td_before"]), reverse=True)[:10]
    expected = None if live_record is None else {
        name: value for name, value in live_record.items()
        if name not in ("game", "local_update", "elapsed_seconds")
    }
    mismatch = None if expected is None else _differences(expected, update_summary)
    # The incident captures a pre-step summary, whose loss was computed in
    # double precision for inspection; the final trace uses the actual loss.
    pre_expected = {name: value for name, value in payload["measurement"].items() if name != "loss"}
    pre_actual = _matching_fields(pre_expected, update_summary)
    pre_mismatch = _differences(pre_expected, pre_actual)
    return {
        "incident": str(path.resolve()), "sha256": _sha256(path),
        "game": payload["game"], "local_updates_before": payload["local_updates"],
        "step_before": payload["optimizer_steps"], "step_after": result.step,
        "offline_games": 0, "offline_updates": 1,
        "live_trace_available": live_record is not None,
        "live_trace_exact_match": None if mismatch is None else not mismatch,
        "live_trace_differences": mismatch,
        "saved_pre_step_measurements_match_except_inspection_loss": not pre_mismatch,
        "saved_pre_step_differences": pre_mismatch,
        "replayed_update": update_summary,
        "probe_before": before_summary, "probe_after": after_summary,
        "probe_change": _probe_changes(before_summary, after_summary, before_zero, after_zero,
                                       before_probe_q, after_probe_q, bank["masks"]),
        "largest_max_legal_q_cases": top_q, "largest_selected_td_error_cases": top_td,
        "all_batch_rows": rows,
        "source_size_and_mtime_unchanged":
            (source_stat.st_size, source_stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns),
        "interpretation_limit": "One reproduced update shows its immediate effect. Crossing a diagnostic Q threshold is not itself evidence that this update caused longer-term representation loss.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="diagnostic run directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    bank_path = Path(manifest["probe_bank"])
    bank = load_probe_bank(bank_path)
    observer_path = run / "instrumentation/updates.py"
    observer_class = _load_observer(observer_path)
    incident_paths = sorted((run / "incidents").glob("*.pt"))
    steps = [int(path.stem.rsplit("_", 1)[-1]) for path in incident_paths]
    live_records = _read_live_records(run / "updates.jsonl", steps)
    analysed = []
    for path, step in zip(incident_paths, steps):
        record = analyse_incident(path, bank=bank, observer_class=observer_class,
                                  live_record=live_records.get(step), device=args.device)
        analysed.append(record)
        top = record["largest_max_legal_q_cases"][0]
        print(json.dumps({"incident": path.name, "step": step,
                          "live_trace_exact_match": record["live_trace_exact_match"],
                          "largest_q": top["max_legal_q_before"],
                          "largest_q_is_recorded_action": top["recorded_action_is_greedy"],
                          "largest_q_row_selected_td": top["selected_td_before"],
                          "last_layer_lost_units": record["probe_change"]["hidden_layers"][-1]["units_that_stopped_activating"]}), flush=True)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "device": args.device, "torch_version": str(torch.__version__),
        "probe_bank": str(bank_path), "probe_bank_sha256": _sha256(bank_path),
        "observer_source": str(observer_path), "observer_sha256": _sha256(observer_path),
        "games_run": 0, "optimizer_updates_replayed": len(analysed),
        "incidents": analysed,
    }
    output = args.output or run / "analysis/incident_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    print("REPORT", output.resolve(), flush=True)
    return int(any(record["live_trace_exact_match"] is False for record in analysed))


if __name__ == "__main__":
    raise SystemExit(main())
