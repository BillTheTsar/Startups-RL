"""Run, evaluate and resume the first DQN experiment.

Nothing runs at import time. The CLI defaults to the agreed 10,000-game run;
tests and experiments can supply a smaller TrainingConfig. Recovery snapshots
are saved between games and around evaluation commits, always replacing latest.
"""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
from random import Random
import tempfile
from typing import Callable
from uuid import uuid4

import torch

from players.neural import FlatObservationEncoder, NeuralPlayer
from players.catalogue import DEFAULT_WEIGHTS_ROOT, ModelSpec, load_pool, restore_pool, snapshot_pool
from players.rule_based.registry import OpponentSpec, RULE_BASED_OPPONENTS

from .archive import append_evaluation_block, get_evaluation_block
from .checkpoints import load_recovery, save_best, save_recovery
from .config import TrainingConfig
from .dqn import DQNLearner
from .episodes import collect_episode, create_episode
from .evaluation import EvaluationSummary, evaluate_checkpoint
from .opponent_pool import OpponentPool
from .q_monitor import collect_q_monitor, monitoring_seed, validate_q_monitor
from .replay import load_initial_buffers
from .replay_refresh import prepare_archive_refresh, refresh_buffers, validate_refresh_snapshot
from .rewards import REWARD_SCHEME, TIMEOUT_REWARD
from .runtime import configure_cpu_threads, restore_cpu_threads_on_error


DEFAULT_ARCHIVE = Path("buffer_data/ranking_player_15.sqlite3")
ProgressCallback = Callable[[dict], None]


def _atomic_json(path: Path, value: dict) -> None:
    """Replace a small JSON report/journal without exposing a partial document."""
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def model_fingerprint(network) -> str:
    """Hash exact named model tensors to identify a frozen evaluation policy."""
    digest = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode("utf-8"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tuples(value):
    """Restore Random's nested tuple state from a JSON-compatible journal."""
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def _opponent_signature(opponents) -> list[dict]:
    """Record resolved settings and neural identities, preserving legacy records."""
    opponents = tuple(opponents)
    if not opponents:
        raise ValueError("opponent pool cannot be empty")
    if any(isinstance(item, ModelSpec) for item in opponents):
        return snapshot_pool(opponents)
    def encode(value):
        if isinstance(value, Fraction):
            return {"numerator": value.numerator, "denominator": value.denominator}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        return value

    signature = []
    for item in opponents:
        if not isinstance(item, OpponentSpec):
            raise ValueError("opponents must contain OpponentSpec objects")
        controller = item.create(1, seed=0)
        weights, parameters = getattr(controller, "weights", None), getattr(controller, "parameters", None)
        signature.append({
            "name": item.name, "class": item.player_class.__module__ + "." + item.player_class.__qualname__,
            "weights": asdict(weights) if weights is not None else None,
            "parameters": asdict(parameters) if parameters is not None else None,
            "chicken": getattr(controller, "chicken", False),
        })
    return json.loads(json.dumps(encode(signature), allow_nan=False))


def _restore_opponents(records, *, device="cpu"):
    """Rebuild the recorded pool without consulting a mutable selection file.

    Old recovery files keep their original rule-based record shape. Restore
    known rule classes from their saved settings; custom Python opponents can
    still be supplied explicitly to resume, as before this catalogue support.
    """
    if not isinstance(records, list) or not records or any(not isinstance(row, dict) for row in records):
        raise ValueError("invalid saved opponent pool")
    if "model_id" in records[0]:
        return restore_pool(records, device=device)
    if records == _opponent_signature(RULE_BASED_OPPONENTS):
        return RULE_BASED_OPPONENTS
    fields = {"name", "class", "weights", "parameters", "chicken"}
    if any(set(row) != fields or not isinstance(row["name"], str) for row in records):
        raise ValueError("invalid legacy opponent records")
    normalized = [{"model_id": row["name"], "kind": "rule_based", "config": row} for row in records]
    try:
        models = restore_pool(normalized, device=device)
    except ValueError as error:
        raise ValueError("cannot recreate the saved custom rule pool; pass the original opponents explicitly") from error
    opponents = tuple(model.factory.__self__ for model in models)
    if _opponent_signature(opponents) != records:
        raise ValueError("restored rule settings differ from the saved opponent pool")
    return opponents


class TrainingRun:
    """Own the training schedule, metrics and persistence for one experiment.

    Build with create or resume, then await run. An optional until_games limit
    makes a recoverable short segment without changing the experiment schedule.
    Evaluations use an independent RNG, frozen online weights and no optimizer.
    Ordinary training episodes enter RAM non_demo only; qualified evaluation
    episodes enter both their RAM pools and the persistent archive.
    """

    @classmethod
    @restore_cpu_threads_on_error
    def create(cls, config: TrainingConfig | None = None, *, archive_path=DEFAULT_ARCHIVE,
               output_dir=None, opponents=RULE_BASED_OPPONENTS,
               progress: ProgressCallback | None = None, max_consecutive_caps: int = 100,
               weights_root=DEFAULT_WEIGHTS_ROOT, opponent_device="cpu"):
        """Load seed replay once, initialize a learner, and save initial recovery.

        Never overwrite an existing run directory or modify the seed archive
        during initialization. Insufficient replay fails before a game starts.
        """
        config = TrainingConfig() if config is None else config
        if not isinstance(config, TrainingConfig):
            raise ValueError("config must be a TrainingConfig")
        configure_cpu_threads(config.cpu_threads)
        if type(max_consecutive_caps) is not int or max_consecutive_caps <= 0:
            raise ValueError("max_consecutive_caps must be a positive integer")
        opponents = tuple(opponents)
        opponent_signature = _opponent_signature(opponents)
        if progress is not None and not callable(progress):
            raise ValueError("progress must be callable or None")
        archive_path = Path(archive_path).resolve()
        buffers = load_initial_buffers(archive_path, config)
        if len(buffers.demo) < config.demo_batch_size or len(buffers.non_demo) < config.non_demo_batch_size:
            raise ValueError("the archive must supply enough demo and non_demo entries for one batch")
        run_id = uuid4().hex
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output_dir = Path("results/training") / f"{stamp}_{run_id[:12]}"
        directory = Path(output_dir).resolve()
        if directory.exists():
            raise FileExistsError("output directory already exists; use resume or choose a new directory")
        learner = DQNLearner(config)
        rng = Random(config.seed)
        state = {
            "version": 3, "run_id": run_id, "archive_path": str(archive_path),
            "reward_scheme": REWARD_SCHEME, "reward_scheme_start_game": 0,
            "weights_root": str(Path(weights_root).resolve()),
            "admissions": [], "refreshes": [], "last_refresh_game": 0,
            "opponents": opponent_signature, "completed_games": 0,
            "attempted_games": 0, "consecutive_caps": 0,
            "max_consecutive_caps": max_consecutive_caps,
            "last_evaluation_game": 0, "promoted_blocks": [],
            "best_mean": None, "best_game": None,
            "games": [], "discarded_attempts": [], "evaluations": [],
            "training_summaries": [], "status": "ready",
        }
        instance = cls._assemble(learner, buffers, state,
                                 {"training": Random(rng.getrandbits(64)),
                                  "evaluation": Random(rng.getrandbits(64)),
                                  "refresh": Random(rng.getrandbits(64))},
                                 directory, opponents, progress, opponent_device=opponent_device)
        directory.mkdir(parents=True, exist_ok=False)
        instance._save()
        return instance

    @classmethod
    def resume(cls, checkpoint_path, *, device=None, cpu_threads=None, opponents=None, opponent_device="cpu",
               progress: ProgressCallback | None = None):
        """Restore networks, optimizer, FIFO replay, RNG streams and schedule.

        Do not reload or resample the archive into RAM. Recreate the recorded
        opponent pool, including neural identities and hashes, unless the same
        pool is supplied explicitly. Other library exports are not added on
        resume; growing mode can still admit future evaluated candidates.
        Opponent inference defaults to CPU; opponent_device may override it.
        A changed device is permitted, without promising cross-device bitwise
        reproducibility. Pending evaluation commits are reconciled by run().
        cpu_threads may override this process's saved PyTorch CPU thread count;
        it does not restart any learning schedule.
        """
        return cls._restore_checkpoint(checkpoint_path, device=device, cpu_threads=cpu_threads, opponents=opponents,
                                       opponent_device=opponent_device, progress=progress)

    @classmethod
    @restore_cpu_threads_on_error
    def _restore_checkpoint(cls, checkpoint_path, *, device=None, cpu_threads=None, opponents=None,
                            opponent_device="cpu", progress=None, replacement_opponents=None):
        """Restore saved training components for resume or a new continuation.

        Only continuation supplies replacement_opponents. In that case the old
        opponent weights need not be loaded: they will not participate in the
        new run. An explicit legacy opponents argument still verifies the source
        pool. The caller must create the child's new state before saving it.
        A deferred replacement-pool factory is resolved only after recovery
        applies the effective CPU thread setting.
        """
        recovered = load_recovery(checkpoint_path, device=device, cpu_threads=cpu_threads)
        if callable(replacement_opponents):
            replacement_opponents = tuple(replacement_opponents())
        state = recovered.run_state
        required = {"version", "run_id", "archive_path", "opponents", "completed_games",
                    "attempted_games", "consecutive_caps", "max_consecutive_caps",
                    "last_evaluation_game", "promoted_blocks", "best_mean", "best_game",
                    "games", "discarded_attempts", "evaluations", "training_summaries", "status"}
        version = state.get("version")
        extension = {"weights_root", "admissions", "refreshes", "last_refresh_game"}
        expected = required if version == 1 else required | extension
        if version == 3:
            expected |= {"reward_scheme", "reward_scheme_start_game"}
        if type(version) is not int or version not in (1, 2, 3) or set(state) not in (expected, expected | {"parent"}):
            raise ValueError("unsupported training-run recovery state")
        if version == 1:
            # These settings were inert in the original runner, including when
            # it saved a version-2 config. Exact resume keeps that behavior;
            # a new continuation can explicitly enable the new schedules.
            recovered.learner.config = replace(recovered.learner.config,
                                                opponent_pool_mode="fixed",
                                                replay_refresh_interval_games=None)
            # Derive a new, independent stream without advancing any old RNG.
            if set(recovered.rngs) != {"training", "evaluation"}:
                raise ValueError("legacy training and evaluation RNG streams are required")
            seed = hashlib.sha256(repr(recovered.rngs["training"].getstate()).encode()).digest()
            recovered.rngs["refresh"] = Random(int.from_bytes(seed, "big"))
            state.update(version=2, weights_root=str(DEFAULT_WEIGHTS_ROOT.resolve()),
                         admissions=[], refreshes=[], last_refresh_game=0)
        if version < 3:
            # Historical rows and a durable pending evaluation keep their old
            # meaning. Future collection uses the newly agreed timeout outcome.
            state.update(version=3, reward_scheme=REWARD_SCHEME,
                         reward_scheme_start_game=state["completed_games"])
        if "parent" in state:
            parent = state["parent"]
            parent_fields = {"source_run_id", "checkpoint_path", "completed_games", "optimizer_steps", "rng_mode"}
            if not isinstance(parent, dict) or set(parent) != parent_fields or (
                any(not isinstance(parent[key], str) or not parent[key]
                    for key in ("source_run_id", "checkpoint_path"))
                or any(type(parent[key]) is not int or parent[key] < 0
                       for key in ("completed_games", "optimizer_steps"))
                or parent["optimizer_steps"] > recovered.learner.optimizer_steps
                or parent["rng_mode"] not in ("inherited", "reseeded")
            ):
                raise ValueError("invalid parent-run metadata")
        if opponents is not None or replacement_opponents is None:
            opponents = (_restore_opponents(state["opponents"], device=opponent_device)
                         if opponents is None else tuple(opponents))
            if state["opponents"] != _opponent_signature(opponents):
                raise ValueError("opponent registry differs from the saved training run")
        if replacement_opponents is not None:
            opponents = tuple(replacement_opponents)
            _opponent_signature(opponents)
        for field in ("completed_games", "attempted_games", "consecutive_caps", "last_evaluation_game"):
            if type(state[field]) is not int or state[field] < 0:
                raise ValueError("invalid saved training counters")
        if type(state["max_consecutive_caps"]) is not int or state["max_consecutive_caps"] <= 0 or (
            state["last_evaluation_game"] > state["completed_games"]
            or state["completed_games"] > recovered.learner.config.training_games
        ):
            raise ValueError("saved counters exceed the configured schedule")
        if state["completed_games"] != len(state["games"]) or state["attempted_games"] != (
            len(state["games"]) + len(state["discarded_attempts"])
        ):
            raise ValueError("training counters do not match saved metrics")
        if set(recovered.rngs) != {"training", "evaluation", "refresh"}:
            raise ValueError("training, evaluation and refresh RNG streams are required")
        cls._validate_extension(state, recovered.learner.config)
        if not Path(state["archive_path"]).is_file():
            raise FileNotFoundError("the run's transition archive is missing")
        if progress is not None and not callable(progress):
            raise ValueError("progress must be callable or None")
        return cls._assemble(recovered.learner, recovered.buffers, state, recovered.rngs,
                             Path(checkpoint_path).resolve().parent, opponents, progress,
                             opponent_device=opponent_device)

    @classmethod
    def _assemble(cls, learner, buffers, state, rngs, directory, opponents, progress, *, opponent_device="cpu"):
        # The pool validates the inference device and supported persistent
        # identities before any growing run is saved. Fixed custom pools retain
        # their legacy representation and behavior.
        if learner.config.opponent_pool_mode == "growing":
            pool = OpponentPool(opponents, learner.config, run_id=state["run_id"],
                                weights_root=state["weights_root"], device=opponent_device)
            opponents = pool.opponents
            state["opponents"] = pool.snapshot()
        instance = cls()
        instance.learner, instance.buffers = learner, buffers
        instance.state, instance.rngs = state, rngs
        instance.directory, instance.opponents = directory, opponents
        instance.progress = progress
        instance.opponent_device = opponent_device
        instance._failed = False
        return instance

    @staticmethod
    def _validate_extension(state, config):
        """Check saved membership/refresh bookkeeping before scheduling new work."""
        if state["reward_scheme"] != REWARD_SCHEME or type(state["reward_scheme_start_game"]) is not int or (
            not 0 <= state["reward_scheme_start_game"] <= state["completed_games"]
        ):
            raise ValueError("invalid saved reward scheme")
        if not isinstance(state["weights_root"], str) or not Path(state["weights_root"]).is_absolute():
            raise ValueError("saved weights_root must be an absolute path")
        last = state["last_refresh_game"]
        if type(last) is not int or not 0 <= last <= state["completed_games"]:
            raise ValueError("invalid last_refresh_game")
        for name in ("admissions", "refreshes"):
            if not isinstance(state[name], list) or any(not isinstance(row, dict) for row in state[name]):
                raise ValueError(f"invalid saved {name}")
        previous = 0
        seen = set()
        for row in state["admissions"]:
            fields = {"run_id", "completed_games", "model_id", "artifact_path", "mean_reward", "threshold", "evaluated_pool"}
            if set(row) != fields or row["run_id"] != state["run_id"] or (
                type(row["completed_games"]) is not int
                or not previous < row["completed_games"] <= state["last_evaluation_game"]
                or type(row["model_id"]) is not str or row["model_id"] in seen
                or row["threshold"] != config.opponent_admission_threshold
                or type(row["mean_reward"]) not in (int, float)
                or not isfinite(row["mean_reward"]) or not row["threshold"] < row["mean_reward"] <= 1.5
                or not isinstance(row["evaluated_pool"], list)
            ):
                raise ValueError("invalid saved admission record")
            previous = row["completed_games"]
            seen.add(row["model_id"])
        if state["admissions"]:
            pool = state["opponents"]
            if config.opponent_pool_mode != "growing" or len(pool) <= len(state["admissions"]):
                raise ValueError("saved admissions do not match the current pool")
            initial_size = len(pool) - len(state["admissions"])
            for index, row in enumerate(state["admissions"], start=initial_size):
                if row["evaluated_pool"] != pool[:index] or row["model_id"] != pool[index].get("model_id") or (
                    row["artifact_path"] != pool[index].get("config", {}).get("artifact_path")
                ):
                    raise ValueError("saved admission history differs from the current pool")
        previous = 0
        for row in state["refreshes"]:
            fields = {"game", "sampling_seed", "archive_snapshot", "before_buffer_sizes", "buffer_sizes"}
            if set(row) not in (fields, fields | {"skipped_invalid"}) or type(row["game"]) is not int or (
                not previous < row["game"] <= state["completed_games"]
                or config.replay_refresh_interval_games is None
                or row["game"] % config.replay_refresh_interval_games
                or type(row["sampling_seed"]) is not int or not 0 <= row["sampling_seed"] < 2**64
            ):
                raise ValueError("invalid saved refresh record")
            validate_refresh_snapshot(row["archive_snapshot"], state["archive_path"])
            if row["archive_snapshot"]["version"] == 2:
                skipped = row.get("skipped_invalid")
                sizes = row["buffer_sizes"]
                partitions = {"demo", "non_demo"}
                if type(skipped) is not dict or set(skipped) != partitions or (
                    type(sizes) is not dict or set(sizes) != partitions
                ):
                    raise ValueError("invalid saved refresh skip counts")
                for name in partitions:
                    selected = min(row["archive_snapshot"]["partitions"][name]["row_count"],
                                   getattr(config, f"{name}_capacity"))
                    if type(skipped[name]) is not int or not 0 <= skipped[name] <= selected or (
                        type(sizes[name]) is not int or sizes[name] != selected - skipped[name]
                        or sizes[name] < getattr(config, f"{name}_batch_size")
                    ):
                        raise ValueError("saved refresh sizes disagree with skipped records")
            elif set(row) != fields:
                raise ValueError("invalid legacy refresh record")
            previous = row["game"]
        if last != previous:
            raise ValueError("last_refresh_game differs from saved refresh history")

    @property
    def config(self) -> TrainingConfig:
        return self.learner.config

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.pt"

    def _report(self, event: dict) -> None:
        if self.progress is not None:
            self.progress(event)

    def _buffer_sizes(self) -> dict[str, int]:
        """Snapshot actual retained transitions in each disjoint RAM pool."""
        return {"demo": len(self.buffers.demo), "non_demo": len(self.buffers.non_demo)}

    def _save(self) -> None:
        """Atomically replace recovery, then derive the readable metrics report."""
        save_recovery(self.latest_path, self.learner, self.buffers,
                      run_state=self.state, rngs=self.rngs)
        _atomic_json(self.directory / "metrics.json", {
            "run_id": self.state["run_id"], "config": self.config.to_dict(),
            "opponents": self.state["opponents"],
            "weights_root": self.state["weights_root"],
            "reward_scheme": self.state["reward_scheme"],
            "reward_scheme_start_game": self.state["reward_scheme_start_game"],
            "admissions": self.state["admissions"], "refreshes": self.state["refreshes"],
            "status": self.state["status"], "completed_games": self.state["completed_games"],
            "optimizer_steps": self.learner.optimizer_steps,
            "buffer_sizes": self._buffer_sizes(),
            "best_mean": self.state["best_mean"], "best_game": self.state["best_game"],
            "games": self.state["games"], "discarded_attempts": self.state["discarded_attempts"],
            "training_summaries": self.state["training_summaries"],
            "evaluations": self.state["evaluations"],
            **({"parent": self.state["parent"]} if "parent" in self.state else {}),
        })

    async def _training_game(self) -> None:
        epsilon = self.config.epsilon_for_game(self.state["completed_games"])

        def factory(player_id, *, seed):
            return NeuralPlayer(player_id, self.learner.online.predict, FlatObservationEncoder(),
                                selection_mode="epsilon_greedy", epsilon=epsilon, seed=seed)

        episode = create_episode(factory, rng=self.rngs["training"], opponents=self.opponents)
        losses = []

        async def update(_transition):
            losses.append(self.learner.update(self.buffers.sample_array_batch()).loss)
            # Permit cancellation between updates; the last durable checkpoint
            # remains the recovery source if this incomplete game is interrupted.
            await asyncio.sleep(0)

        self.state["attempted_games"] += 1
        result = await collect_episode(episode, max_actions=self.config.max_actions_per_game,
                                       on_transition=update)
        self.buffers.non_demo.extend(result.transitions)
        self.state["completed_games"] += 1
        self.state["consecutive_caps"] = 0
        self.state["games"].append({
            "game": self.state["completed_games"], "attempt": self.state["attempted_games"],
            "reward": result.outcome.reward, "outright_win": result.outcome.outright_win,
            "capped": result.capped, "reward_scheme": REWARD_SCHEME,
            "epsilon": epsilon, "actions": result.action_count, "transitions": len(result.transitions),
            "updates": len(losses), "optimizer_steps": self.learner.optimizer_steps,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "buffer_sizes": self._buffer_sizes(),
        })

    def _evaluation_due(self) -> bool:
        games = self.state["completed_games"]
        return games > 0 and games % self.config.evaluation_interval_games == 0 and (
            self.state["last_evaluation_game"] < games
        )

    def _check_evaluation_metadata(self, metadata: dict, block_id: int, fingerprint: str) -> None:
        """Validate the durable result before promotion or best-model selection."""
        required = {"version", "run_id", "block_id", "model_fingerprint", "summary", "evaluation_rng_after"}
        if not isinstance(metadata, dict):
            raise ValueError("incompatible pending evaluation metadata")
        version = metadata.get("version")
        expected = required if version == 1 else required | {"evaluated_opponents"}
        if version in (3, 4):
            expected |= {"reward_scheme"}
        if set(metadata) != expected or (
            type(version) is not int or version not in (1, 2, 3, 4)
            or type(metadata["block_id"]) is not int or metadata["block_id"] != block_id
            or metadata["model_fingerprint"] != fingerprint or metadata["run_id"] != self.state["run_id"]
        ):
            raise ValueError("incompatible pending evaluation metadata")
        if version >= 2 and metadata["evaluated_opponents"] != self.state["opponents"]:
            raise ValueError("pending evaluation used a different opponent pool")
        if version >= 3 and metadata["reward_scheme"] != REWARD_SCHEME:
            raise ValueError("pending evaluation uses an unknown reward scheme")
        summary = metadata["summary"]
        keys = {"game", "attempted_games", "completed_games", "capped_games", "rewards", "mean_reward",
                "valid", "qualifies", "outright_wins"}
        if version >= 3:
            keys |= {"capped_mask", "trajectory_games"}
        if version == 4:
            keys.add("q_monitor")
        if not isinstance(summary, dict) or set(summary) != keys:
            raise ValueError("invalid evaluation summary")
        if (version == 4) != (self.config.q_monitor_sample_size is not None):
            raise ValueError("evaluation monitoring does not match the configuration")
        if version == 4:
            validate_q_monitor(summary["q_monitor"], self.config)
            if summary["q_monitor"]["sampling_seed"] != monitoring_seed(self.state["run_id"], block_id):
                raise ValueError("evaluation monitoring seed does not match this boundary")
        for field in ("game", "attempted_games", "completed_games", "capped_games", "outright_wins"):
            if type(summary[field]) is not int or summary[field] < 0:
                raise ValueError("invalid evaluation counts")
        rewards = summary["rewards"]
        if not isinstance(rewards, list) or any(type(reward) not in (int, float) or
                                               not isfinite(reward) or not -1.5 <= reward <= 1.5
                                               for reward in rewards):
            raise ValueError("invalid evaluation rewards")
        mean = sum(rewards) / len(rewards) if rewards else None
        valid = summary["capped_games"] == 0 if version < 3 else True
        if version >= 3:
            flags = summary["capped_mask"]
            if not isinstance(flags, list) or len(flags) != len(rewards) or (
                any(type(flag) is not bool for flag in flags)
                or sum(flags) != summary["capped_games"]
                or any(flag and reward != TIMEOUT_REWARD for flag, reward in zip(flags, rewards))
                or type(summary["trajectory_games"]) is not int
                or not 0 <= summary["trajectory_games"] <= len(rewards)
            ):
                raise ValueError("invalid evaluation timeout flags or trajectory count")
        qualifies = valid and mean is not None and mean > self.config.promotion_threshold
        if summary["game"] != block_id or summary["attempted_games"] != self.config.evaluation_games or (
            summary["completed_games"] != len(rewards)
            or summary["completed_games"] + (summary["capped_games"] if version < 3 else 0) != summary["attempted_games"]
            or summary["outright_wins"] != sum(reward == 1.5 for reward in rewards)
            or summary["mean_reward"] != mean or summary["valid"] is not valid
            or summary["qualifies"] is not qualifies
        ):
            raise ValueError("evaluation result does not match its counts/rewards/configuration")
        Random().setstate(_tuples(metadata["evaluation_rng_after"]))

    async def _evaluation(self) -> None:
        """Commit one evaluation, recovering safely after any interrupted stage.

        Save the exact pre-evaluation state. Qualified episodes and their RNG
        continuation commit atomically to SQLite. A small temporary JSON journal
        also retains nonqualifying results before best/recovery files change.
        Resume reuses either durable result, then applies its RAM insertion once.
        """
        self._save()
        block_id = self.state["completed_games"]
        # Keep the same immutable candidate for evaluation, export and best.pt.
        # Recovery reconstructs it from the pre-evaluation checkpoint.
        frozen = deepcopy(self.learner.online).eval().requires_grad_(False)
        frozen.zero_grad(set_to_none=True)
        fingerprint = model_fingerprint(frozen)
        evaluated_opponents = deepcopy(self.state["opponents"])
        evaluation_pool = tuple(self.opponents)
        journal_path = self.directory / "pending_evaluation.json"
        archived = get_evaluation_block(self.state["archive_path"],
                                        run_id=self.state["run_id"], block_id=block_id)
        metadata = None
        if journal_path.exists():
            candidate = json.loads(journal_path.read_text(encoding="utf-8"))
            if candidate.get("run_id") != self.state["run_id"]:
                raise ValueError("pending evaluation belongs to another run")
            if candidate.get("block_id") == block_id:
                metadata = candidate
            elif candidate.get("block_id", block_id) > self.state["last_evaluation_game"]:
                raise ValueError("pending evaluation does not match recovery boundary")
        if archived is not None:
            if archived.model_fingerprint != fingerprint:
                raise ValueError("committed evaluation does not match the recovered online model")
            if metadata is not None and metadata != archived.metadata:
                raise ValueError("evaluation journal and archive disagree")
            metadata = archived.metadata
        if metadata is None:
            monitor = None
            if self.config.q_monitor_sample_size is not None:
                monitor = collect_q_monitor(
                    self.learner, self.buffers, sample_size=self.config.q_monitor_sample_size,
                    seed=monitoring_seed(self.state["run_id"], block_id),
                )
            result = await evaluate_checkpoint(frozen, self.config,
                                                rng=self.rngs["evaluation"], opponents=evaluation_pool)
            summary = {
                "game": block_id, "attempted_games": result.attempted_games,
                "completed_games": len(result.episodes), "capped_games": result.capped_games,
                "rewards": list(result.rewards), "mean_reward": result.mean_reward,
                "valid": result.valid, "qualifies": result.qualifies,
                "outright_wins": sum(item.outcome.outright_win for item in result.episodes),
                "capped_mask": [item.capped for item in result.episodes],
                "trajectory_games": sum(bool(item.transitions) for item in result.episodes),
            }
            if monitor is not None:
                summary["q_monitor"] = monitor
            # JSON canonicalization gives the same value after either the
            # journal or SQLite metadata round trip (Random uses nested tuples).
            metadata = json.loads(json.dumps({
                "version": 4 if monitor is not None else 3,
                "run_id": self.state["run_id"], "block_id": block_id,
                "reward_scheme": REWARD_SCHEME,
                "model_fingerprint": fingerprint, "summary": summary,
                "evaluation_rng_after": self.rngs["evaluation"].getstate(),
                "evaluated_opponents": evaluated_opponents,
            }, allow_nan=False))
            if result.qualifies and result.partitioned_episodes():
                archived = append_evaluation_block(
                    self.state["archive_path"], run_id=self.state["run_id"], block_id=block_id,
                    model_fingerprint=fingerprint, episodes=result.partitioned_episodes(), metadata=metadata,
                )
        self._check_evaluation_metadata(metadata, block_id, fingerprint)
        summary = dict(metadata["summary"])
        evaluated_opponents = deepcopy(metadata.get("evaluated_opponents", evaluated_opponents))
        if summary["qualifies"] and summary.get("trajectory_games", 1) > 0 and archived is None:
            raise ValueError("qualified evaluation is missing its committed archive block")
        if archived is not None and not summary["qualifies"]:
            raise ValueError("a nonqualifying evaluation cannot have an archive commit")
        if archived is not None and metadata["version"] >= 3 and len(archived.episodes) != summary["trajectory_games"]:
            raise ValueError("archive trajectory count differs from the evaluation result")
        _atomic_json(journal_path, metadata)
        self.rngs["evaluation"].setstate(_tuples(metadata["evaluation_rng_after"]))
        if archived is not None:
            if block_id in self.state["promoted_blocks"]:
                raise ValueError("evaluation was already applied to live replay")
            for partition, transitions in archived.episodes:
                getattr(self.buffers, partition).extend(transitions)
            self.state["promoted_blocks"].append(block_id)
        admission = {"qualified": False, "added": False, "reason": "fixed_pool",
                     "model_id": None, "artifact_path": None}
        if self.config.opponent_pool_mode == "growing":
            pool = OpponentPool(self.opponents, self.config, run_id=self.state["run_id"],
                                weights_root=self.state["weights_root"], device=self.opponent_device)
            decision = pool.consider_candidate(
                frozen, EvaluationSummary(tuple(summary["rewards"]), summary["attempted_games"],
                                           summary["capped_games"], summary["mean_reward"],
                                           tuple(summary["capped_mask"]) if "capped_mask" in summary else None),
                completed_games=block_id, evaluated_pool=evaluated_opponents,
            )
            admission = asdict(decision)
            admission["artifact_path"] = str(decision.artifact_path) if decision.artifact_path is not None else None
            self.opponents = pool.opponents
            self.state["opponents"] = pool.snapshot()
            self.state["admissions"].extend(record.to_dict() for record in pool.admissions)
            improved = decision.qualified  # Latest qualifier, even at a lower score.
        else:
            improved = summary["valid"] and (self.state["best_mean"] is None or
                                             summary["mean_reward"] > self.state["best_mean"])
        summary["improved_best"] = improved
        summary["evaluated_opponents"] = evaluated_opponents
        summary["admission"] = admission
        summary["pool_size_after"] = len(self.opponents)
        summary["reward_scheme"] = metadata.get("reward_scheme", "placement_points_v1")
        summary["buffer_sizes"] = self._buffer_sizes()
        if improved:
            save_best(self.directory / "best.pt", frozen, config=self.config,
                      metadata={"run_id": self.state["run_id"], "game": block_id,
                                "mean_reward": summary["mean_reward"], "model_fingerprint": fingerprint,
                                "reward_scheme": summary["reward_scheme"],
                                "opponents": evaluated_opponents,
                                "best_selection": "latest_qualifying" if self.config.opponent_pool_mode == "growing" else "highest_mean"})
            self.state["best_mean"], self.state["best_game"] = summary["mean_reward"], block_id
        self.state["evaluations"].append(summary)
        self.state["last_evaluation_game"] = block_id
        self._save()
        journal_path.unlink(missing_ok=True)
        self._report({"type": "evaluation", **summary})

    def _refresh_due(self) -> bool:
        interval = self.config.replay_refresh_interval_games
        games = self.state["completed_games"]
        return interval is not None and games > 0 and games % interval == 0 and self.state["last_refresh_game"] < games

    def _refresh(self) -> None:
        """Replace both buffers once, with retryable bounds and a dedicated seed.

        Save before preparing the refresh. A journal fixes the archive prefixes
        and seed before any replacement. On retry, later archive appends cannot
        change the selected rows. The post-refresh checkpoint commits both RAM
        buffers, their sampling RNGs, the refresh RNG and its progress together.
        """
        self._save()
        game = self.state["completed_games"]
        path = self.directory / "pending_refresh.json"
        rng_before = self.rngs["refresh"].getstate()
        next_rng = Random()
        next_rng.setstate(rng_before)
        sampling_seed = next_rng.getrandbits(64)
        if path.exists():
            metadata = json.loads(path.read_text(encoding="utf-8"))
        else:
            metadata = {
                "version": 1, "run_id": self.state["run_id"], "game": game,
                "sampling_seed": sampling_seed, "rng_before": rng_before,
                "archive_snapshot": prepare_archive_refresh(self.state["archive_path"]),
            }
            _atomic_json(path, metadata)
        fields = {"version", "run_id", "game", "sampling_seed", "rng_before", "archive_snapshot"}
        if not isinstance(metadata, dict) or set(metadata) != fields or (
            type(metadata["version"]) is not int or metadata["version"] != 1
            or metadata["run_id"] != self.state["run_id"]
            or type(metadata["game"]) is not int or metadata["game"] != game
            or type(metadata["sampling_seed"]) is not int or metadata["sampling_seed"] != sampling_seed
            or _tuples(metadata["rng_before"]) != rng_before
        ):
            raise ValueError("pending refresh does not match the recovery boundary")
        before = self._buffer_sizes()
        replacement = refresh_buffers(self.state["archive_path"], self.config,
                                      rng=Random(sampling_seed), snapshot=metadata["archive_snapshot"])
        self.buffers = replacement
        self.rngs["refresh"].setstate(next_rng.getstate())
        self.state["last_refresh_game"] = game
        record = {"game": game, "sampling_seed": sampling_seed,
                  "archive_snapshot": metadata["archive_snapshot"],
                  "before_buffer_sizes": before, "buffer_sizes": self._buffer_sizes()}
        if metadata["archive_snapshot"]["version"] == 2:
            record["skipped_invalid"] = {
                name: min(metadata["archive_snapshot"]["partitions"][name]["row_count"],
                          getattr(self.config, f"{name}_capacity")) - len(getattr(replacement, name))
                for name in ("demo", "non_demo")
            }
        self.state["refreshes"].append(record)
        self._save()
        path.unlink(missing_ok=True)
        self._report({"type": "replay_refresh", **record})

    def _clean_committed_journals(self) -> None:
        """Remove journals left by a crash after their recovery save succeeded."""
        for filename, field, counter in (
            ("pending_evaluation.json", "block_id", "last_evaluation_game"),
            ("pending_refresh.json", "game", "last_refresh_game"),
        ):
            path = self.directory / filename
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("run_id") != self.state["run_id"] or (
                type(record.get(field)) is not int
            ):
                raise ValueError("pending journal does not belong to this run")
            if record[field] <= self.state[counter]:
                path.unlink()
            elif record[field] != self.state["completed_games"]:
                raise ValueError("pending journal does not match the recovery boundary")
            elif (field == "block_id" and not self._evaluation_due()) or (
                field == "game" and not self._refresh_due()
            ):
                raise ValueError("pending journal is outside its configured schedule")

    async def run(self, *, until_games: int | None = None) -> dict:
        """Train to the configured total (or an earlier recoverable segment end).

        Count natural and timeout training endings, not evaluation games.
        Timeout trajectories enter non_demo with reward -1. Unexpected
        errors/interruption preserve the prior durable checkpoint; discard this
        in-memory session and resume from disk rather than reusing partial state.
        """
        if self._failed:
            raise RuntimeError("this session failed; resume from its recovery checkpoint")
        limit = self.config.training_games if until_games is None else until_games
        if type(limit) is not int or not self.state["completed_games"] <= limit <= self.config.training_games:
            raise ValueError("until_games must be between the completed and configured game totals")
        configure_cpu_threads(self.config.cpu_threads)
        if self.state["status"] == "stopped_after_caps":
            # An explicit retry gets a fresh cap budget, retaining prior attempts.
            self.state["consecutive_caps"] = 0
        self.state["status"] = "running"
        try:
            self._clean_committed_journals()
            if self._evaluation_due():
                await self._evaluation()
            if self._refresh_due():
                self._refresh()
            while self.state["completed_games"] < limit:
                previous_games = self.state["completed_games"]
                await self._training_game()
                if self.state["completed_games"] == previous_games:
                    continue
                games = self.state["completed_games"]
                if games % self.config.evaluation_interval_games == 0:
                    window = self.state["games"][-self.config.evaluation_interval_games:]
                    record = {"game": games, "games": len(window),
                              "mean_reward": sum(item["reward"] for item in window) / len(window),
                              "epsilon": window[-1]["epsilon"], "optimizer_steps": self.learner.optimizer_steps,
                              "buffer_sizes": self._buffer_sizes()}
                    self.state["training_summaries"].append(record)
                    self._report({"type": "training_summary", **record})
                if self._evaluation_due():
                    await self._evaluation()
                if self._refresh_due():
                    self._refresh()
                elif games % self.config.checkpoint_interval_games == 0 and self.state["last_evaluation_game"] != games:
                    self._save()
            self.state["status"] = "complete" if limit == self.config.training_games else "paused"
            self._save()
            self._report({"type": "finished", **self.summary()})
            return self.summary()
        except BaseException:
            self._failed = True
            raise

    def summary(self) -> dict:
        """Return small run-level results; detailed per-game metrics live on disk."""
        return {"run_id": self.state["run_id"], "status": self.state["status"],
                "completed_games": self.state["completed_games"],
                "discarded_games": len(self.state["discarded_attempts"]),
                "capped_games": sum(row.get("capped", False) for row in self.state["games"]),
                "optimizer_steps": self.learner.optimizer_steps,
                "buffer_sizes": self._buffer_sizes(),
                "best_mean": self.state["best_mean"], "best_game": self.state["best_game"],
                "opponent_count": len(self.opponents), "admission_count": len(self.state["admissions"]),
                "refresh_count": len(self.state["refreshes"]),
                "output_dir": str(self.directory)}


def _print_progress(event: dict) -> None:
    kind = event["type"]
    if kind == "training_summary":
        print(f"Training {event['game']}: mean reward {event['mean_reward']:.4f}, "
              f"epsilon {event['epsilon']:.4f}, updates {event['optimizer_steps']}", flush=True)
    elif kind == "evaluation":
        mean = "n/a" if event["mean_reward"] is None else f"{event['mean_reward']:.4f}"
        print(f"Evaluation at {event['game']}: mean {mean}, capped {event['capped_games']}, "
              f"best updated={event['improved_best']}, promoted={event['qualifies']}, "
              f"opponent added={event['admission']['added']}, pool={event['pool_size_after']}", flush=True)
        if "q_monitor" in event:
            monitored = event["q_monitor"]
            def number(value):
                return "n/a" if value is None else f"{value:.4g}"
            values = monitored["online"]["max_legal_q"]
            target = monitored["target"]["max_legal_q"]
            outside = values.get("outside_return_range_fraction")
            outside_text = "n/a" if outside is None else f"{outside:.1%}"
            print(f"Q monitor ({monitored['sample_size']} states): online best legal Q "
                  f"median={number(values.get('median'))}, p95={number(values.get('p95'))}, "
                  f"p99={number(values.get('p99'))}, max={number(values.get('max'))}, "
                  f"outside [-1.5, 1.5]={outside_text}, nonfinite={values['nonfinite_count']}; "
                  f"target p99={number(target.get('p99'))}", flush=True)
    elif kind == "replay_refresh":
        skipped = event.get("skipped_invalid", {"demo": 0, "non_demo": 0})
        print(f"Replay refreshed at {event['game']}: demo {event['buffer_sizes']['demo']}, "
              f"non_demo {event['buffer_sizes']['non_demo']}; skipped invalid: "
              f"demo {skipped['demo']}, non_demo {skipped['non_demo']}", flush=True)
    elif kind == "cap_stop":
        print(f"Stopped after {event['consecutive_caps']} consecutive capped games; recovery saved.", flush=True)
    elif kind == "finished":
        print(f"{event['status']}: {event['completed_games']} games, {event['optimizer_steps']} updates. "
              f"Files: {event['output_dir']}", flush=True)


def main(argv=None) -> int:
    """CLI for a fresh run or exact recovery; no implicit full training on import."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help=f"seed/archive database (default: {DEFAULT_ARCHIVE})")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, help="JSON produced by TrainingConfig.to_dict()")
    parser.add_argument("--games", type=int, help="total training games for a fresh run")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--cpu-threads", type=int, help="PyTorch CPU threads; overrides saved/configured count (default: 1)")
    parser.add_argument("--catalog-file", type=Path, help="ordered pool JSON for a fresh run; defaults to rule-based opponents")
    parser.add_argument("--opponent-device", default="cpu", help="device for frozen neural opponents (default: cpu)")
    parser.add_argument("--weights-root", type=Path, help="destination for newly admitted neural players (default: players/neural/weights)")
    parser.add_argument("--resume", type=Path, help="path to latest.pt")
    parser.add_argument("--until-games", type=int, help="save and stop at this completed-game count")
    args = parser.parse_args(argv)
    if args.resume and any(value is not None for value in (args.config, args.games, args.seed, args.output, args.archive, args.catalog_file, args.weights_root)):
        parser.error("resume uses saved config, opponent pool and run directory; only device/cpu-threads/opponent-device/until-games may override")
    try:
        if args.resume:
            run = TrainingRun.resume(args.resume, device=args.device, cpu_threads=args.cpu_threads,
                                     opponent_device=args.opponent_device, progress=_print_progress)
        else:
            config = TrainingConfig() if args.config is None else TrainingConfig.from_dict(
                json.loads(args.config.read_text(encoding="utf-8")))
            overrides = {field: value for field, value in (
                ("training_games", args.games), ("seed", args.seed), ("device", args.device),
                ("cpu_threads", args.cpu_threads)
            ) if value is not None}
            config = replace(config, **overrides)
            configure_cpu_threads(config.cpu_threads)
            run = TrainingRun.create(config, archive_path=args.archive or DEFAULT_ARCHIVE,
                                     output_dir=args.output, progress=_print_progress,
                                     weights_root=args.weights_root or DEFAULT_WEIGHTS_ROOT,
                                     opponent_device=args.opponent_device,
                                     opponents=RULE_BASED_OPPONENTS if args.catalog_file is None else
                                     load_pool(args.catalog_file, device=args.opponent_device))
        print(f"Run {run.state['run_id']} on {run.learner.device}, algorithm={run.config.algorithm}, "
              f"architecture=v{run.config.architecture_version}, "
              f"clip_targets={run.config.clip_targets}, "
              f"CPU threads={run.config.cpu_threads}; "
              f"recovery: {run.latest_path}", flush=True)
        result = asyncio.run(run.run(until_games=args.until_games))
        return 2 if result["status"] == "stopped_after_caps" else 0
    except KeyboardInterrupt:
        print("Interrupted. Resume from the last saved latest.pt checkpoint.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
