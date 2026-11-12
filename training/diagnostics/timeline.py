"""Compare surviving checkpoints on one fixed, stratified observation bank.

Run with ``python -m training.diagnostics.timeline``. This is a read-only
analysis of training sources and writes only to its diagnostic output directory.
It does not start games, train models, load a shared archive, or export opponents.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from random import Random

import numpy as np
import torch

from players.neural.artifacts import load_neural_artifact, neural_policy_id
from training.checkpoints import _BEST_FIELDS, _RECOVERY_FIELDS, _model, _read_weights_only, _validate_payload
from training.config import TrainingConfig
from .probes import load_probe_bank, measure_network


ROOT = Path(__file__).resolve().parents[2]
EARLY = ROOT / "results/training/20261105T060816Z_8060854903a6/latest.pt"
LATE = ROOT / "results/training/20261106T001851Z_e9f8a65f5914/latest.pt"


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def read_checkpoint(path):
    """Hash and deserialize the same file handle, including a live latest.pt."""
    path = Path(path).resolve()
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        payload = _read_weights_only(stream)
    kind = payload.get("kind") if isinstance(payload, dict) else None
    if kind not in ("best", "recovery"):
        raise ValueError(f"unsupported diagnostic checkpoint: {path}")
    _validate_payload(payload, kind, _BEST_FIELDS if kind == "best" else _RECOVERY_FIELDS)
    TrainingConfig.from_dict(payload["config"])
    return payload, checksum


def build_probe_bank(sources, output, *, seed=650394, per_stratum=512):
    """Sample early/late x demo/non_demo x take/play without replacement."""
    rng = Random(seed)
    state_rows, mask_rows, labels, indices = [], [], [], []
    manifest = {"seed": seed, "per_stratum": per_stratum, "sources": [], "strata": []}
    for age, path, payload, checksum in sources:
        if payload["kind"] != "recovery":
            raise ValueError("probe source must contain recovery replay")
        run = payload["run_state"]
        manifest["sources"].append({"label": age, "path": str(path.resolve()),
                                    "sha256": checksum, "run_id": run["run_id"],
                                    "game": run["completed_games"],
                                    "optimizer_steps": payload["optimizer_steps"]})
        for partition in ("demo", "non_demo"):
            replay = payload["buffers"][partition]
            if replay["byte_order"] != "little":
                raise ValueError("unsupported replay byte order")
            states = np.frombuffer(replay["states"], dtype="<f4").reshape(replay["size"], 332)
            raw_masks = np.frombuffer(replay["masks"], dtype=np.uint8).reshape(replay["size"], 41)
            for phase_index, phase in enumerate(("take", "play")):
                eligible = np.flatnonzero(states[:, phase_index] == 1).tolist()
                if len(eligible) < per_stratum:
                    raise ValueError(f"not enough {age}/{partition}/{phase} states")
                selected = np.array(sorted(rng.sample(eligible, per_stratum)), dtype=np.int64)
                if not np.isin(raw_masks[selected], (0, 1)).all():
                    raise ValueError("invalid probe mask bytes")
                label = f"{age}/{partition}/{phase}"
                state_rows.append(states[selected].copy())
                mask_rows.append(raw_masks[selected].astype(np.bool_))
                labels.extend([label] * len(selected))
                indices.extend(selected.tolist())
                manifest["strata"].append({"label": label, "eligible": len(eligible),
                                            "selected": len(selected)})
    np.savez_compressed(output / "probe_bank.npz", states=np.concatenate(state_rows),
                        masks=np.concatenate(mask_rows), strata=np.asarray(labels),
                        row_indices=np.asarray(indices, dtype=np.int64))
    manifest["bank_sha256"] = hashlib.sha256((output / "probe_bank.npz").read_bytes()).hexdigest()
    manifest["n_states"] = len(indices)
    _json(output / "probe_bank.json", manifest)
    return manifest


def _run_index():
    result = {}
    for path in sorted((ROOT / "results/training").glob("*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result[data["run_id"]] = {
            "directory": str(path.parent.resolve()), "parent": data.get("parent"),
            "completed_games": data["completed_games"],
            "config": data["config"],
            "updates_by_game": {row["game"]: row["optimizer_steps"] for row in data["games"]},
            "evaluations": [{"game": row["game"], "mean_reward": row["mean_reward"],
                              "capped_games": row["capped_games"]} for row in data["evaluations"]],
        }
    return result


def _source_metadata(payload, path, checksum):
    kind = payload["kind"]
    state = payload["run_state"] if kind == "recovery" else payload["metadata"]
    return {
        "type": "checkpoint", "path": str(path.resolve()), "sha256": checksum,
        "kind": kind, "run_id": state.get("run_id"),
        "game": state.get("completed_games", state.get("game")),
        "optimizer_steps": payload.get("optimizer_steps"),
        "saved_evaluation_mean": state.get("mean_reward"),
        "parent": state.get("parent"),
    }


def _artifact_source(artifact):
    provenance = artifact.manifest["provenance"]
    # Exports contain source_metadata; automatic admissions carry fields directly.
    state = provenance.get("source_metadata", provenance.get("metadata", provenance))
    return {
        "type": "artifact", "path": str(artifact.path), "name": artifact.name,
        "model_id": artifact.model_id, "run_id": state.get("run_id"),
        "game": state.get("completed_games", state.get("game")),
        "optimizer_steps": state.get("optimizer_steps"),
        "saved_evaluation_mean": state.get("mean_reward"),
        "provenance": provenance,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics/20261106/timeline")
    parser.add_argument("--early", type=Path, default=EARLY)
    parser.add_argument("--late", type=Path, default=LATE)
    parser.add_argument("--seed", type=int, default=650394)
    parser.add_argument("--per-stratum", type=int, default=512)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    torch.set_num_threads(1)
    early, early_hash = read_checkpoint(args.early)
    late, late_hash = read_checkpoint(args.late)
    sources = [("early", args.early, early, early_hash), ("late", args.late, late, late_hash)]
    bank_metadata = build_probe_bank(sources, output, seed=args.seed, per_stratum=args.per_stratum)
    bank = load_probe_bank(output / "probe_bank.npz")
    print("PROBE_BANK", str(output / "probe_bank.npz"), bank_metadata["n_states"], flush=True)
    runs = _run_index()
    records = {}

    def analyze(network, source, *, target=None, config=None):
        identity = neural_policy_id(network)
        run = runs.get(source.get("run_id"))
        if run and source.get("optimizer_steps") is None:
            source["optimizer_steps"] = run["updates_by_game"].get(source.get("game"))
        if run:
            source["parent"] = run["parent"]
        if identity not in records:
            online_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
            torch.save({"format": "diagnostic_network_v1", "online": online_state,
                        "model_id": identity, "source": source, "config": config},
                       output / "models" / f"{identity}.pt")
            records[identity] = {"model_id": identity, "sources": [], "online": measure_network(
                network, bank["states"], bank["masks"], strata=bank["strata"]), "targets": []}
        records[identity]["sources"].append(source)
        if target is not None:
            target_id = neural_policy_id(target)
            target_path = output / "models" / f"target_{identity}_{target_id}.pt"
            torch.save({"format": "diagnostic_network_v1", "target": target.state_dict(),
                        "online_model_id": identity, "target_model_id": target_id,
                        "source": source}, target_path)
            records[identity]["targets"].append({"model_id": target_id, "source": source,
                "snapshot": str(target_path), "measurements": measure_network(
                    target, bank["states"], bank["masks"], strata=bank["strata"])})
        row = records[identity]["online"]
        print("MODEL", source.get("run_id"), source.get("game"), identity[-10:],
              "qmax", row["max_legal_q"]["max"], "last_active", row["hidden_layers"][-1]["ever_active_units"],
              "zero", row["hidden_layers"][-1]["all_zero_fraction"], flush=True)

    cached = {args.early.resolve(): (early, early_hash), args.late.resolve(): (late, late_hash)}
    for path in sorted((ROOT / "results/training").glob("*/*.pt")):
        payload, checksum = cached.get(path.resolve(), (None, None))
        if payload is None:
            payload, checksum = read_checkpoint(path)
        online = _model(payload["online"], payload["model_dtype"], payload["architecture_version"]).eval().requires_grad_(False)
        target = None if payload["kind"] != "recovery" else _model(
            payload["target"], payload["model_dtype"], payload["architecture_version"]).eval().requires_grad_(False)
        analyze(online, _source_metadata(payload, path, checksum), target=target, config=payload["config"])
    # Permanent admission exports supply intermediate points absent from latest.pt.
    for path in sorted((ROOT / "players/neural/weights").glob("neural_*")):
        artifact = load_neural_artifact(path)
        analyze(artifact.online, _artifact_source(artifact))
    for run in runs.values():
        run.pop("updates_by_game")
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "probe_bank": bank_metadata,
              "runs": runs, "models": list(records.values()),
              "limitations": ["Only surviving weights can be measured; missing historical checkpoints cannot be reconstructed.",
                              "Best and admitted models are performance-selected snapshots, not an unbiased timeline.",
                              "Inactivity and Q statistics describe this fixed replay sample, not all possible states."]}
    _json(output / "timeline.json", result)
    with (output / "timeline.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("run_id", "game", "optimizer_steps", "model_id", "kind", "source", "q_median", "q_p99",
                  "q_max", "q_above_1_5_fraction", "last_ever_active", "last_all_zero_fraction")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records.values():
            stats = row["online"]
            for source in row["sources"]:
                writer.writerow(dict(zip(fields, (
                    source.get("run_id"), source.get("game"), source.get("optimizer_steps"), row["model_id"],
                    source.get("kind", "artifact"), source["path"], stats["max_legal_q"]["median"],
                    stats["max_legal_q"]["p99"], stats["max_legal_q"]["max"],
                    stats["max_legal_q"]["above_1_5_fraction"], stats["hidden_layers"][-1]["ever_active_units"],
                    stats["hidden_layers"][-1]["all_zero_fraction"]))))
    print("TIMELINE_COMPLETE", len(records), "unique models", flush=True)


if __name__ == "__main__":
    main()
