"""Observe unchanged DQN updates in an isolated, bounded training experiment.

This is a diagnostic continuation, not an exact historical replay. It inherits
the online/target networks, AdamW state, replay and random streams from a full
checkpoint, but keeps its selected opponents fixed and does not evaluate,
promote demonstrations, admit opponents, refresh replay, or write an archive.
The original update and episode collector remain responsible for learning.
All measurements and checkpoints are written inside a new output directory.
"""

import argparse
import asyncio
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from random import Random
import time
import traceback

import torch

from players.catalogue import load_pool, snapshot_pool
from players.neural import FlatObservationEncoder, NeuralPlayer
from training.batches import ArrayBatch
from training.checkpoints import _atomic_save, _cpu_copy, load_recovery, save_recovery
from training.config import TrainingConfig
from training.episodes import collect_episode, create_episode
from training.replay import reconfigure_buffers
from training.runtime import configure_cpu_threads

from .probes import load_probe_bank, measure_network
from .updates import UpdateObserver


MAX_DIAGNOSTIC_GAMES = 5_000


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _write_json(path: Path, value) -> None:
    """Replace a small JSON report atomically without leaving half a report."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _json_line(stream, record) -> None:
    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def _global_rng() -> dict:
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def _weights_payload(learner, *, local_games: int, local_updates: int, kind: str) -> dict:
    """Capture a small learning snapshot; its payload intentionally omits replay."""
    return {
        "diagnostic_version": 1, "kind": kind,
        "local_games": local_games, "local_updates": local_updates,
        "optimizer_steps": learner.optimizer_steps,
        "config": learner.config.to_dict(),
        "online": _cpu_copy(learner.online.state_dict()),
        "target": _cpu_copy(learner.target.state_dict()),
        "optimizer": _cpu_copy(learner.optimizer.state_dict()),
        "global_rng": _global_rng(),
    }


async def run_diagnostic(
    *, checkpoint: str | Path, config: TrainingConfig, opponents,
    probe_bank: str | Path, output: str | Path, games: int = MAX_DIAGNOSTIC_GAMES,
    probe_interval: int = 100, progress_every: int = 100,
    snapshot_interval: int = 5_000, max_incidents: int = 10,
) -> dict:
    """Run at most 5,000 games while measuring each original optimizer update.

    The diagnostic continuation uses the requested configuration's learning,
    exploration, batch and cap settings. It preserves saved random streams when
    the seed matches; a changed seed follows ordinary continuation reseeding.
    Measurement calls never draw from those streams. Opponents are immutable
    specifications throughout this experiment. No production run or database is
    modified. Refuse existing output directories to prevent accidental reuse.

    Initial/final recovery files are readable with load_recovery, but their
    diagnostic run_state is not a normal TrainingRun resume state. Small weight
    snapshots and incident files are labelled separately and are not normal
    recovery checkpoints. Incident files include the exact batch and optimizer
    state before the observed update, enabling that update to be reproduced.
    """
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    for name, value in (("games", games), ("probe_interval", probe_interval),
                        ("progress_every", progress_every), ("snapshot_interval", snapshot_interval)):
        _positive(value, name)
    if games > MAX_DIAGNOSTIC_GAMES:
        raise ValueError("diagnostic game budget cannot exceed 5000")
    if type(max_incidents) is not int or max_incidents < 0:
        raise ValueError("max_incidents must be a nonnegative integer")
    output, checkpoint, probe_bank = Path(output).resolve(), Path(checkpoint).resolve(), Path(probe_bank).resolve()
    if output.exists():
        raise FileExistsError("diagnostic output already exists; choose a new directory")
    opponents = tuple(opponents)
    pool_snapshot = snapshot_pool(opponents)
    bank = load_probe_bank(probe_bank)
    configure_cpu_threads(config.cpu_threads)
    # Hash and load through one open file: a concurrent atomic replacement can
    # never associate one checkpoint's identity with another one's tensors.
    with checkpoint.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        recovered = load_recovery(stream, device=config.device, cpu_threads=config.cpu_threads)
    original = recovered.learner.config
    if config.gamma != original.gamma:
        raise ValueError("gamma must remain unchanged for this diagnostic continuation")
    learner = recovered.learner
    buffers = reconfigure_buffers(recovered.buffers, config)
    learner.config = config
    for group in learner.optimizer.param_groups:
        group["lr"], group["weight_decay"] = config.learning_rate, config.weight_decay
    learner.optimizer.defaults.update(lr=config.learning_rate, weight_decay=config.weight_decay)
    rngs = recovered.rngs
    reseeded = config.seed != original.seed
    if reseeded:
        rng = Random(config.seed)
        rngs = {"training": Random(rng.getrandbits(64)), "evaluation": Random(rng.getrandbits(64))}
        buffers.demo.reseed(rng.getrandbits(64))
        buffers.non_demo.reseed(rng.getrandbits(64))
        random.seed(rng.getrandbits(64))
        torch.manual_seed(rng.getrandbits(64))
        rngs["refresh"] = Random(rng.getrandbits(64))
    elif "refresh" not in rngs:
        digest = hashlib.sha256(repr(rngs["training"].getstate()).encode()).digest()
        rngs["refresh"] = Random(int.from_bytes(digest, "big"))
    if "training" not in rngs:
        raise ValueError("source checkpoint must contain the training random stream")

    start_steps = learner.optimizer_steps
    context = {"game": 0, "completed_games": 0, "updates": 0}
    experiment_id = output.name
    manifest = {
        "diagnostic_version": 1, "experiment_id": experiment_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint), "source_sha256": source_hash,
        "source_run_id": recovered.run_state.get("run_id"),
        "source_completed_games": recovered.run_state.get("completed_games"),
        "source_optimizer_steps": start_steps,
        "config": config.to_dict(), "opponents": pool_snapshot,
        "probe_bank": str(probe_bank),
        "probe_bank_sha256": hashlib.sha256(probe_bank.read_bytes()).hexdigest(),
        "probe_interval_updates": probe_interval, "snapshot_interval_updates": snapshot_interval,
        "game_budget": games, "rng_mode": "reseeded" if reseeded else "inherited",
        "initial_buffer_sizes": {"demo": len(buffers.demo), "non_demo": len(buffers.non_demo)},
        "controlled_differences": [
            "Fixed opponent pool throughout this diagnostic run.",
            "No evaluation, demonstration promotion, opponent admission or archive writes.",
            "No scheduled archive replacement; completed training episodes enter non_demo normally.",
            "Extra deterministic measurements and diagnostic snapshots only; DQN update is unchanged.",
        ],
        "recovery_note": "initial_recovery.pt and final_recovery.pt have diagnostic run_state; use load_recovery, not training.train --resume.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "snapshots").mkdir()
    (output / "incidents").mkdir()
    _write_json(output / "manifest.json", manifest)

    def recovery_state(status: str) -> dict:
        return {
            "kind": "diagnostic_update_run", "version": 1, "run_id": experiment_id,
            "completed_games": context["completed_games"], "local_updates": context["updates"],
            "source_checkpoint": str(checkpoint), "source_sha256": source_hash,
            "opponents": pool_snapshot, "status": status,
        }

    save_recovery(output / "initial_recovery.pt", learner, buffers,
                  run_state=recovery_state("initial"), rngs=rngs)
    wall_start = time.perf_counter()
    incident_records = []
    last_incident_step = -10_000
    incident_peak = 0.0
    latest_probe = None
    last_probe_step = -1
    game_records = []

    with ExitStack() as stack:
        updates_file = stack.enter_context((output / "updates.jsonl").open("w", encoding="utf-8", buffering=1024 * 1024))
        games_file = stack.enter_context((output / "games.jsonl").open("w", encoding="utf-8", buffering=64 * 1024))
        probes_file = stack.enter_context((output / "probes.jsonl").open("w", encoding="utf-8", buffering=1024 * 1024))

        def probe(reason: str):
            nonlocal latest_probe, last_probe_step
            detailed = reason in ("initial", "final")
            strata = bank.get("strata") if detailed else None
            record = {
                "game": context["game"], "completed_games": context["completed_games"],
                "local_updates": context["updates"], "optimizer_steps": learner.optimizer_steps,
                "reason": reason, "elapsed_seconds": time.perf_counter() - wall_start,
                "online": measure_network(learner.online, bank["states"], bank["masks"], strata=strata),
                "target": measure_network(learner.target, bank["states"], bank["masks"], strata=strata)
                if detailed or learner.optimizer_steps % config.target_update_interval == 0 else None,
            }
            _json_line(probes_file, record)
            probes_file.flush()
            latest_probe, last_probe_step = record, learner.optimizer_steps
            return record

        def incident_before_step(observed_learner, batch, summary):
            nonlocal last_incident_step, incident_peak
            severity = max(value for value in (
                summary["max_legal_q"]["abs_max"], summary["abs_td_error"]["max"], 0.0,
            ) if value is not None)
            gradient = summary["gradient_l2"]
            nonfinite = summary["gradient_nonfinite_count"] > 0 or (
                summary["max_legal_q"]["nonfinite_count"] > 0
            )
            finite_gradient = 0.0 if gradient is None else gradient
            suspicious = severity >= 5.0 or finite_gradient >= 10.0 or nonfinite
            step = summary["step"]
            novelty = float("inf") if nonfinite else max(severity / 5.0, finite_gradient / 10.0)
            if not suspicious or len(incident_records) >= max_incidents or step - last_incident_step < 1_000:
                return
            # Avoid spending all incident files on ten almost identical updates.
            if incident_records and novelty < incident_peak * 1.5:
                return
            number = len(incident_records) + 1
            name = f"incident_{number:02d}_step_{step}.pt"
            payload = _weights_payload(observed_learner, local_games=context["completed_games"],
                                       local_updates=context["updates"], kind="diagnostic_before_update")
            payload.update(
                game=context["game"], measurement=summary,
                batch={name: torch.from_numpy(getattr(batch, name).copy()) for name in ArrayBatch.FIELD_NAMES},
            )
            _atomic_save(output / "incidents" / name, payload)
            incident_records.append({"file": name, "game": context["game"], "step": step,
                                     "severity": severity, "gradient_l2": gradient})
            last_incident_step, incident_peak = step, max(incident_peak, novelty)
            # Model-forward hooks are still installed by the observer here.
            # Preserve the pre-update state; inspect it separately afterward.
            _write_json(output / "incidents.json", incident_records)

        observer = UpdateObserver(before_step=incident_before_step)
        probe("initial")
        pending_incidents = 0
        try:
            for local_game in range(1, games + 1):
                context["game"] = local_game
                epsilon = config.epsilon_for_game(local_game - 1)

                def focal_factory(player_id, *, seed):
                    return NeuralPlayer(player_id, learner.online.predict, FlatObservationEncoder(),
                                        selection_mode="epsilon_greedy", epsilon=epsilon, seed=seed)

                episode = create_episode(focal_factory, rng=rngs["training"], opponents=opponents)
                losses = []

                async def on_transition(_transition):
                    nonlocal pending_incidents
                    batch = buffers.sample_array_batch()
                    result, summary = observer.update(learner, batch)
                    context["updates"] += 1
                    losses.append(result.loss)
                    _json_line(updates_file, {
                        "game": local_game, "local_update": context["updates"],
                        "elapsed_seconds": time.perf_counter() - wall_start, **summary,
                    })
                    if len(incident_records) > pending_incidents:
                        probe("incident_after_update")
                        pending_incidents = len(incident_records)
                    elif result.target_updated:
                        probe("target_update")
                    elif context["updates"] % probe_interval == 0:
                        probe("interval")
                    if context["updates"] % snapshot_interval == 0:
                        _atomic_save(output / "snapshots" / f"update_{context['updates']:08d}.pt",
                                     _weights_payload(learner, local_games=context["completed_games"],
                                                      local_updates=context["updates"], kind="diagnostic_weights"))
                    await asyncio.sleep(0)

                result = await collect_episode(episode, max_actions=config.max_actions_per_game,
                                               on_transition=on_transition)
                buffers.non_demo.extend(result.transitions)
                context["completed_games"] = local_game
                game_record = {
                    "game": local_game, "epsilon": epsilon, "reward": result.outcome.reward,
                    "capped": result.capped, "actions": result.action_count,
                    "transitions": len(result.transitions), "updates": len(losses),
                    "optimizer_steps": learner.optimizer_steps,
                    "mean_loss": sum(losses) / len(losses) if losses else None,
                    "buffer_sizes": {"demo": len(buffers.demo), "non_demo": len(buffers.non_demo)},
                    "elapsed_seconds": time.perf_counter() - wall_start,
                }
                game_records.append(game_record)
                _json_line(games_file, game_record)
                if local_game % progress_every == 0 or local_game == games:
                    updates_file.flush()
                    games_file.flush()
                    progress = {
                        **game_record,
                        "probe_max_legal_q": latest_probe["online"]["max_legal_q"]["max"],
                        "probe_active_last_units": latest_probe["online"]["hidden_layers"][-1]["ever_active_units"],
                        "probe_last_zero_fraction": latest_probe["online"]["hidden_layers"][-1]["all_zero_fraction"],
                        "incidents_saved": len(incident_records),
                    }
                    _write_json(output / "progress.json", progress)
                    print("DIAGNOSTIC_PROGRESS", json.dumps(progress, separators=(",", ":")), flush=True)
            probe("final")
            save_recovery(output / "final_recovery.pt", learner, buffers,
                          run_state=recovery_state("complete"), rngs=rngs)
            summary = {
                "status": "complete", "completed_games": context["completed_games"],
                "optimizer_updates": context["updates"], "source_optimizer_steps": start_steps,
                "final_optimizer_steps": learner.optimizer_steps,
                "elapsed_seconds": time.perf_counter() - wall_start,
                "capped_games": sum(row["capped"] for row in game_records),
                "mean_training_reward": sum(row["reward"] for row in game_records) / games,
                "final_buffer_sizes": {"demo": len(buffers.demo), "non_demo": len(buffers.non_demo)},
                "incidents": incident_records, "final_probe": latest_probe,
            }
            _write_json(output / "summary.json", summary)
            print("DIAGNOSTIC_COMPLETE", json.dumps({k: v for k, v in summary.items() if k not in ("final_probe", "incidents")}), flush=True)
            return summary
        except BaseException as error:
            updates_file.flush()
            games_file.flush()
            probes_file.flush()
            failure = {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                **context, "optimizer_steps": learner.optimizer_steps,
                "error": repr(error), "traceback": traceback.format_exc(),
                "note": "An unfinished game may have applied updates; this is not a resumable game-boundary state.",
            }
            try:
                save_recovery(output / "failure_recovery.pt", learner, buffers,
                              run_state=recovery_state("failed_or_interrupted_mid_game"), rngs=rngs)
            except Exception as save_error:
                failure["recovery_save_error"] = repr(save_error)
                _atomic_save(output / "failure_weights.pt", _weights_payload(
                    learner, local_games=context["completed_games"],
                    local_updates=context["updates"], kind="diagnostic_failure_weights"))
            _write_json(output / "failure.json", failure)
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog-file", type=Path, required=True)
    parser.add_argument("--probe-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=MAX_DIAGNOSTIC_GAMES)
    parser.add_argument("--probe-interval", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--snapshot-interval", type=int, default=5_000)
    parser.add_argument("--max-incidents", type=int, default=10)
    args = parser.parse_args(argv)
    if not 0 < args.games <= MAX_DIAGNOSTIC_GAMES:
        parser.error("--games must be between 1 and 5000")
    config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    configure_cpu_threads(config.cpu_threads)
    opponents = load_pool(args.catalog_file, device="cpu")
    asyncio.run(run_diagnostic(
        checkpoint=args.checkpoint, config=config, opponents=opponents,
        probe_bank=args.probe_bank, output=args.output, games=args.games,
        probe_interval=args.probe_interval, progress_every=args.progress_every,
        snapshot_interval=args.snapshot_interval, max_incidents=args.max_incidents,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
