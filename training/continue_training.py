"""Start a new training run from a full recovery checkpoint.

Unlike exact resume, continuation creates a new directory and starts new game,
epsilon, evaluation and saving schedules. Online/target weights, AdamW history
and its lifetime update count, and every replay example are inherited. The
source archive is shared; setup never reloads it into replay or changes it.
Nothing runs at import time.
"""

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from random import Random
from uuid import uuid4

import torch

from players.catalogue import load_pool

from .config import TrainingConfig
from .replay import reconfigure_buffers
from .runtime import configure_cpu_threads
from .train import ProgressCallback, TrainingRun, _opponent_signature, _print_progress


def create_continuation(
    checkpoint_path, config: TrainingConfig | None = None, *, output_dir=None,
    seed: int | None = None, device: str | None = None, games: int | None = None,
    cpu_threads: int | None = None,
    opponents=None, new_opponents=None, opponent_device="cpu", weights_root=None,
    progress: ProgressCallback | None = None,
) -> TrainingRun:
    """Copy saved training state into a new run, with optional changed settings.

    Use the source settings when config is omitted; otherwise config is a full
    replacement. Explicit seed/device/games/cpu_threads arguments take priority. Gamma must
    stay unchanged, capacities must fit all inherited entries, and each positive
    batch size must fit its pool's actual occupancy. Inherit the source opponent
    pool unless new_opponents explicitly supplies a replacement for the new run.
    new_opponents may also be a zero-argument factory, resolved after the saved
    or overridden CPU thread count is applied and before loading old opponents.
    The archive path and consecutive-cap limit are retained. The older opponents
    argument, when supplied, verifies the source pool (useful for custom Python
    opponents); it does not change the child pool. opponent_device controls
    loading inherited and newly admitted frozen neural opponents and defaults
    to CPU. weights_root selects where newly admitted opponents are saved;
    omission inherits the parent's destination. Existing opponent files remain
    at their recorded paths even when that destination changes.

    A replacement configuration may change algorithm between dqn and double_dqn.
    Both networks, AdamW moments and replay are retained; the new algorithm is
    used for subsequent training targets. Omission inherits the saved algorithm.
    It may also enable or disable clip_targets without resetting learned state.
    An exact resume retains the saved clipping setting.

    The new run counts games from zero, including epsilon/evaluation/save and
    replay-refresh schedules. Its admission and refresh histories start empty;
    inherited opponents remain available. Target updates still use the inherited
    lifetime optimizer count.
    AdamW moments and per-parameter steps are retained; only its learning rate
    and weight decay settings are replaced after loading.

    With unchanged config.seed and no explicit seed argument, preserve saved
    RNG states. A changed config.seed, or an explicit integer seed (even if it
    equals the saved seed), resets future game, exploration, evaluation, replay
    sampling and global Python/PyTorch randomness. A changed seed of None uses
    system randomness. Neither mode changes model weights or replay contents.

    Copy the checkpoint exactly as saved. Do not finish a parent's pending
    evaluation or modify its files; use exact resume first if that is wanted.
    Return an initialized run with its own initial latest.pt and metrics.json.
    The parent checkpoint is only a source: subsequent child resumes need the
    shared archive, but do not need that parent checkpoint to remain available.
    """
    if config is not None and not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer or None")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    checkpoint_path = Path(checkpoint_path).resolve()
    run_id = uuid4().hex
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("results/training") / f"{stamp}_{run_id[:12]}"
    directory = Path(output_dir).resolve()
    if directory.exists():
        raise FileExistsError("output directory already exists; choose a new directory")

    # Loading a recovery checkpoint restores global RNGs. Keep rejected requests
    # from disturbing the caller's random streams.
    previous_python = random.getstate()
    previous_cpu = torch.get_rng_state().clone()
    previous_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    previous_threads = torch.get_num_threads()
    try:
        requested_device = device if device is not None else config.device if config is not None else None
        requested_threads = cpu_threads if cpu_threads is not None else config.cpu_threads if config is not None else None
        source = TrainingRun._restore_checkpoint(
            checkpoint_path, device=requested_device, cpu_threads=requested_threads, opponents=opponents,
            opponent_device=opponent_device, replacement_opponents=new_opponents,
        )
        selected_opponents = source.opponents
        selected_signature = _opponent_signature(selected_opponents)
        original = source.config
        settings = original if config is None else config
        overrides = {}
        if seed is not None:
            overrides["seed"] = seed
        if device is not None:
            overrides["device"] = device
        if games is not None:
            overrides["training_games"] = games
        if cpu_threads is not None:
            overrides["cpu_threads"] = cpu_threads
        settings = replace(settings, **overrides)
        configure_cpu_threads(settings.cpu_threads)
        if settings.gamma != original.gamma:
            raise ValueError("gamma must remain unchanged when continuing a saved run")
        if settings.architecture_version != original.architecture_version:
            raise ValueError("architecture_version must remain unchanged for continuation; start fresh training for a different architecture")

        buffers = reconfigure_buffers(source.buffers, settings)
        learner = source.learner
        learner.config = settings
        # load_state_dict restored the old parameter-group settings too. Replace
        # them only now, leaving all moment tensors and AdamW step values intact.
        for group in learner.optimizer.param_groups:
            group["lr"] = settings.learning_rate
            group["weight_decay"] = settings.weight_decay
        learner.optimizer.defaults.update(lr=settings.learning_rate, weight_decay=settings.weight_decay)

        reseeded = seed is not None or settings.seed != original.seed
        rngs = source.rngs
        if reseeded:
            rng = Random(settings.seed)
            rngs = {"training": Random(rng.getrandbits(64)), "evaluation": Random(rng.getrandbits(64))}
            buffers.demo.reseed(rng.getrandbits(64))
            buffers.non_demo.reseed(rng.getrandbits(64))
            random.seed(rng.getrandbits(64))
            torch.manual_seed(rng.getrandbits(64))
            # Add the refresh stream after the original seed draws, retaining
            # the earlier reseeding behavior of every existing stream.
            rngs["refresh"] = Random(rng.getrandbits(64))

        state = {
            **source.state,
            "run_id": run_id, "completed_games": 0, "attempted_games": 0,
            "opponents": selected_signature,
            "consecutive_caps": 0, "last_evaluation_game": 0, "promoted_blocks": [],
            "best_mean": None, "best_game": None, "games": [], "discarded_attempts": [],
            "evaluations": [], "training_summaries": [], "status": "ready",
            "admissions": [], "refreshes": [], "last_refresh_game": 0,
            "reward_scheme_start_game": 0,
            "weights_root": source.state["weights_root"] if weights_root is None else
                            str(Path(weights_root).resolve()),
            "parent": {
                "source_run_id": source.state["run_id"], "checkpoint_path": str(checkpoint_path),
                "completed_games": source.state["completed_games"],
                "optimizer_steps": learner.optimizer_steps,
                "rng_mode": "reseeded" if reseeded else "inherited",
            },
        }
        instance = TrainingRun._assemble(
            learner, buffers, state, rngs, directory, selected_opponents, progress,
            opponent_device=opponent_device,
        )
        directory.mkdir(parents=True, exist_ok=False)
        instance._save()
        return instance
    except BaseException:
        configure_cpu_threads(previous_threads)
        random.setstate(previous_python)
        torch.set_rng_state(previous_cpu)
        if previous_cuda:
            torch.cuda.set_rng_state_all(previous_cuda)
        raise


def main(argv=None) -> int:
    """CLI for a new run that inherits a checkpoint's learned state and replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-checkpoint", type=Path, required=True, help="source full recovery checkpoint, usually latest.pt")
    parser.add_argument("--config", type=Path, help="complete TrainingConfig.to_dict() JSON; defaults to source settings")
    parser.add_argument("--output", type=Path, help="new output directory; must not exist")
    parser.add_argument("--games", type=int, help="total games in this new run; overrides config but not epsilon decay")
    parser.add_argument("--seed", type=int, help="reset future RNG streams, even if this matches the source seed")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--cpu-threads", type=int, help="PyTorch CPU threads; overrides config/source count (default: 1)")
    parser.add_argument("--catalog-file", type=Path, help="pool JSON for this new run; omit to inherit the checkpoint's pool")
    parser.add_argument("--opponent-device", default="cpu", help="device for frozen neural opponents (default: cpu)")
    parser.add_argument("--weights-root", type=Path,
                        help="destination for newly admitted neural players; defaults to the parent's saved destination")
    parser.add_argument("--until-games", type=int, help="save and stop at this new run's completed-game count")
    args = parser.parse_args(argv)
    try:
        config = None if args.config is None else TrainingConfig.from_dict(
            json.loads(args.config.read_text(encoding="utf-8")))
        if args.cpu_threads is not None or config is not None:
            configure_cpu_threads(args.cpu_threads if args.cpu_threads is not None else config.cpu_threads)
        run = create_continuation(
            args.from_checkpoint, config, output_dir=args.output, seed=args.seed,
            device=args.device, games=args.games, cpu_threads=args.cpu_threads, progress=_print_progress,
            new_opponents=None if args.catalog_file is None else
                          lambda: load_pool(args.catalog_file, device=args.opponent_device),
            opponent_device=args.opponent_device, weights_root=args.weights_root,
        )
        print(f"New run {run.state['run_id']} from {run.state['parent']['source_run_id']} "
              f"on {run.learner.device}, algorithm={run.config.algorithm}, "
              f"architecture=v{run.config.architecture_version}, "
              f"clip_targets={run.config.clip_targets}, "
              f"CPU threads={run.config.cpu_threads}; recovery: {run.latest_path}", flush=True)
        result = asyncio.run(run.run(until_games=args.until_games))
        return 2 if result["status"] == "stopped_after_caps" else 0
    except KeyboardInterrupt:
        print("Interrupted. Resume the new run from its own latest.pt checkpoint.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
