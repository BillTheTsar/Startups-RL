"""Atomic recovery and inference checkpoints with explicit, validated formats.

Recovery contains everything needed between games: learner, compact replay,
caller-owned run metadata and RNG streams. Best-model files contain inference
weights and metadata only. Neither format writes to the SQLite archive. Reads
use torch.load(weights_only=True); no Python model/controller objects are saved.
The architecture version is checked against the saved configuration before
constructing either the original ReLU model or the LayerNorm version.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
from math import isfinite
import os
from pathlib import Path
import random
from random import Random
import tempfile

import torch
from torch import Tensor

from players.neural.q_network import QNetwork

from .config import TrainingConfig
from .dqn import DQNLearner, _resolve_device
from .replay import ReplayBuffers
from .runtime import configure_cpu_threads, restore_cpu_threads_on_error


CHECKPOINT_VERSION = 1
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32, "float64": torch.float64}
_BEST_FIELDS = {"format_version", "kind", "architecture_version", "model_dtype",
                "config", "online", "metadata"}
_RECOVERY_FIELDS = {"format_version", "kind", "architecture_version", "model_dtype", "config",
                    "online", "target", "optimizer", "optimizer_steps", "buffers", "run_state",
                    "rngs", "global_rng"}


@dataclass(frozen=True)
class RecoveryState:
    """Restored training components; the caller interprets run_state's counters."""

    learner: DQNLearner
    buffers: ReplayBuffers
    run_state: dict[str, object]
    rngs: dict[str, Random]


@dataclass(frozen=True)
class BestCheckpoint:
    """A frozen evaluation-mode model with its configuration and caller metadata."""

    online: QNetwork
    config: TrainingConfig
    metadata: dict[str, object]


@dataclass(frozen=True)
class InferenceCheckpoint:
    """The online model read for export, without restoring any training state.

    online is frozen on CPU. metadata contains the best checkpoint's metadata,
    or selected source-run details for recovery checkpoints. source_sha256
    identifies the exact source file, separately from the exported policy ID.
    """

    online: QNetwork
    config: TrainingConfig
    kind: str
    metadata: dict[str, object]
    source_sha256: str


def _primitives(value, *, field: str):
    """Copy metadata containing only finite primitive values and string-key maps."""
    if value is None or type(value) in (bool, int, str, bytes):
        return value
    if type(value) is float and isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_primitives(item, field=field) for item in value)
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _primitives(item, field=field) for key, item in value.items()}
    raise ValueError(f"{field} must contain only finite primitives and string-key dictionaries")


def _metadata(value, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a dictionary")
    return _primitives(value, field=field)


def _cpu_copy(value):
    """Copy tensor trees into ordinary dictionaries and detached CPU tensors."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return deepcopy(value)


def _dtype_name(network: QNetwork) -> str:
    dtypes = {parameter.dtype for parameter in network.parameters()}
    for name, dtype in _DTYPES.items():
        if dtypes == {dtype}:
            return name
    raise ValueError("checkpoint models must use one supported floating-point dtype")


def _validate_weights(state, dtype_name: str, architecture_version: int = 1) -> None:
    if type(dtype_name) is not str or dtype_name not in _DTYPES:
        raise ValueError("unsupported checkpoint model dtype")
    shapes = QNetwork.parameter_shapes(architecture_version)
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("checkpoint model parameters do not match the architecture")
    for name, shape in shapes.items():
        value = state[name]
        if not isinstance(value, Tensor) or value.layout != torch.strided or (
            tuple(value.shape) != shape or value.dtype != _DTYPES[dtype_name]
        ) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid checkpoint model tensor: {name}")


def _model(state, dtype_name: str, architecture_version: int = 1) -> QNetwork:
    """Rebuild the declared network, preserving the legacy version-1 default."""
    _validate_weights(state, dtype_name, architecture_version)
    # Loading must not consume the caller's global initialization stream.
    with torch.random.fork_rng(devices=[]):
        model = QNetwork(architecture_version=architecture_version).to(dtype=_DTYPES[dtype_name])
    model.load_state_dict(state, strict=True)
    return model


def _validate_optimizer(state, learner: DQNLearner, *, optimizer_steps: int) -> None:
    """Check AdamW parameter mappings and moment shapes before load_state_dict."""
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("invalid checkpoint optimizer fields")
    groups = state["param_groups"]
    expected = torch.optim.AdamW(
        learner.online.parameters(), lr=learner.config.learning_rate,
        weight_decay=learner.config.weight_decay,
    ).state_dict()["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        raise ValueError("checkpoint optimizer must have one parameter group")
    group = _metadata(groups[0], "optimizer parameter group")
    if group != expected[0] or any(type(index) is not int for index in group.get("params", ())):
        raise ValueError("checkpoint AdamW settings do not match the configuration")
    moments = state["state"]
    parameters = tuple(learner.online.parameters())
    if not isinstance(moments, dict) or any(type(index) is not int or not 0 <= index < len(parameters)
                                          for index in moments):
        raise ValueError("invalid checkpoint optimizer parameter IDs")
    expected_ids = set(range(len(parameters))) if optimizer_steps else set()
    if set(moments) != expected_ids:
        raise ValueError("checkpoint AdamW moments are incomplete for its optimizer-step count")
    for index, values in moments.items():
        if not isinstance(values, dict) or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("invalid checkpoint AdamW moment fields")
        step = values["step"]
        if not isinstance(step, Tensor) or step.layout != torch.strided or step.ndim != 0 or (
            not step.is_floating_point() or not bool(torch.isfinite(step))
            or step.item() < 0 or not float(step.item()).is_integer()
        ):
            raise ValueError("invalid checkpoint AdamW step counter")
        if step.item() != optimizer_steps:
            raise ValueError("checkpoint AdamW step counters must match optimizer_steps")
        for name in ("exp_avg", "exp_avg_sq"):
            value = values[name]
            parameter = parameters[index]
            if not isinstance(value, Tensor) or value.layout != torch.strided or (
                value.shape != parameter.shape or value.dtype != parameter.dtype
            ) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid checkpoint AdamW tensor: {name}")
            if name == "exp_avg_sq" and bool((value < 0).any()):
                raise ValueError("AdamW second moments cannot be negative")


def _random_from_state(state, field: str) -> Random:
    rng = Random()
    try:
        rng.setstate(state)
    except (TypeError, ValueError, OverflowError, IndexError) as error:
        raise ValueError(f"invalid {field} RNG state") from error
    return rng


def _validate_rngs(states) -> dict[str, Random]:
    if not isinstance(states, dict) or any(type(name) is not str or not name for name in states):
        raise ValueError("named RNG states must have nonempty string names")
    return {name: _random_from_state(state, name) for name, state in states.items()}


def _validate_global_rng(states) -> None:
    if not isinstance(states, dict) or set(states) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("invalid global RNG checkpoint fields")
    _random_from_state(states["python"], "global Python")
    cpu = states["torch_cpu"]
    if not isinstance(cpu, Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1 or cpu.device.type != "cpu":
        raise ValueError("invalid PyTorch CPU RNG tensor")
    try:
        torch.Generator(device="cpu").set_state(cpu)
    except RuntimeError as error:
        raise ValueError("invalid PyTorch CPU RNG state") from error
    cuda = states["torch_cuda"]
    if not isinstance(cuda, list):
        raise ValueError("CUDA RNG states must be a list")
    for state in cuda:
        if not isinstance(state, Tensor) or state.dtype != torch.uint8 or state.ndim != 1 or (
            state.device.type != "cpu" or not state.numel()
        ):
            raise ValueError("invalid CUDA RNG tensor")
        if torch.cuda.is_available():
            try:
                torch.Generator(device="cuda").set_state(state)
            except RuntimeError as error:
                raise ValueError("invalid CUDA RNG state") from error


def _check_buffers_config(buffers: ReplayBuffers, config: TrainingConfig) -> None:
    if not isinstance(buffers, ReplayBuffers) or (
        buffers.demo.capacity != config.demo_capacity or buffers.non_demo.capacity != config.non_demo_capacity
        or buffers.demo_batch_size != config.demo_batch_size
        or buffers.non_demo_batch_size != config.non_demo_batch_size
    ):
        raise ValueError("replay capacities and batch sizes must match the training configuration")


def _check_replay_snapshot_config(state, config: TrainingConfig) -> None:
    """Reject capacity/batch mismatches before allocating restored replay rings."""
    if not isinstance(state, dict):
        raise ValueError("checkpoint replay state must be a dictionary")
    for name in ("demo", "non_demo"):
        pool = state.get(name)
        batch_size = state.get(f"{name}_batch_size")
        if not isinstance(pool, dict) or type(pool.get("capacity")) is not int or (
            pool["capacity"] != getattr(config, f"{name}_capacity")
            or type(batch_size) is not int or batch_size != getattr(config, f"{name}_batch_size")
        ):
            raise ValueError("saved replay capacities and batch sizes must match the training configuration")


def _atomic_save(path: str | Path, payload: dict[str, object]) -> None:
    """Flush a temporary file beside its destination, then replace atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_weights_only(source) -> dict[str, object]:
    """Deserialize a checkpoint path or open binary stream without restoring it."""
    try:
        # Pickle protocol 2 spells empty bytes as the builtin bytes constructor.
        # PyTorch already accepts nonempty bytes but omits that benign builtin
        # from its default allowlist. Permit only bytes in this scoped context;
        # arbitrary classes/functions remain blocked by weights_only=True.
        with torch.serialization.safe_globals([bytes]):
            return torch.load(source, map_location="cpu", weights_only=True)
    except (FileNotFoundError, PermissionError):
        raise
    except Exception as error:
        raise ValueError("checkpoint is not a readable weights-only file") from error


def _validate_payload(payload, kind: str, fields: set[str]) -> dict[str, object]:
    """Check the common checkpoint header without interpreting training state."""
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("checkpoint fields do not match the requested format")
    if type(payload["format_version"]) is not int or payload["format_version"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if payload["kind"] != kind:
        raise ValueError(f"expected a {kind} checkpoint")
    if type(payload["architecture_version"]) is not int or (
        payload["architecture_version"] not in QNetwork.SUPPORTED_ARCHITECTURE_VERSIONS
    ):
        raise ValueError("unsupported checkpoint network architecture version")
    config = TrainingConfig.from_dict(payload["config"])
    if config.architecture_version != payload["architecture_version"]:
        raise ValueError("checkpoint architecture_version does not match its configuration")
    return payload


def _load(path: str | Path, kind: str, fields: set[str]) -> dict[str, object]:
    return _validate_payload(_read_weights_only(path), kind, fields)


def read_inference_checkpoint(path: str | Path) -> InferenceCheckpoint:
    """Read the online network from a best or recovery file for permanent export.

    Validate the saved format, configuration and online network. Never restore
    RNGs, construct an optimizer/replay buffer, or require the training archive.
    Recovery-only internals are not validated: this is not a recovery check.
    torch.load still reads the full source payload into memory before its
    unrelated fields are discarded, so latest.pt can need more RAM than best.pt.

    Hash and deserialize through the same open file so an atomic checkpoint
    replacement during export cannot mix two source versions. Model construction
    preserves the caller's random-number states. The source is read-only. Both
    supported architecture versions retain their original inference behavior.
    """
    with Path(path).open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        payload = _read_weights_only(stream)
    if not isinstance(payload, dict) or payload.get("kind") not in ("best", "recovery"):
        raise ValueError("expected a best or recovery training checkpoint")
    kind = payload["kind"]
    _validate_payload(payload, kind, _BEST_FIELDS if kind == "best" else _RECOVERY_FIELDS)
    config = TrainingConfig.from_dict(payload["config"])
    if kind == "best":
        metadata = _metadata(payload["metadata"], "metadata")
    else:
        state = payload["run_state"]
        if not isinstance(state, dict):
            raise ValueError("run_state must be a dictionary")
        # Avoid copying per-game metrics, replay or random streams into an
        # opponent. A prior best score is not a score for this online network.
        metadata = _metadata({name: state[name] for name in (
            "run_id", "completed_games", "attempted_games", "parent", "opponents",
            "reward_scheme", "reward_scheme_start_game",
        ) if name in state}, "source run metadata")
        steps = payload["optimizer_steps"]
        if type(steps) is not int or steps < 0:
            raise ValueError("invalid checkpoint optimizer_steps")
        metadata["optimizer_steps"] = steps
    online = _model(payload["online"], payload["model_dtype"], payload["architecture_version"])
    online.eval().requires_grad_(False)
    return InferenceCheckpoint(online, config, kind, metadata, source_sha256)


def save_recovery(
    path: str | Path, learner: DQNLearner, buffers: ReplayBuffers, *,
    run_state: dict[str, object], rngs: dict[str, Random],
) -> None:
    """Atomically replace a complete recovery checkpoint at a game boundary.

    Capture online and target weights, AdamW moments, global optimizer count,
    encoder/codec/config versions, FIFO replay and every supplied RNG stream.
    Include LayerNorm scale/offset and their optimizer moments for version 2.
    The caller supplies primitive counters, metrics, current opponent identities,
    admission and refresh history, and pending archive identifiers in run_state.
    Named streams include the independent refresh RNG. Their schedule validation
    belongs to TrainingRun; this storage format also supports older callers.
    No inference that a checkpoint is best
    is made here. Invalid input or a failed write preserves the previous file.
    """
    if not isinstance(learner, DQNLearner):
        raise ValueError("learner must be a DQNLearner")
    learner.online.validate_architecture()
    learner.target.validate_architecture()
    architecture_version = learner.config.architecture_version
    if learner.online.architecture_version != architecture_version or (
        learner.target.architecture_version != architecture_version
    ):
        raise ValueError("checkpoint networks must match the configured architecture_version")
    _check_buffers_config(buffers, learner.config)
    if type(learner.optimizer_steps) is not int or learner.optimizer_steps < 0:
        raise ValueError("optimizer_steps must be a nonnegative integer")
    run_state = _metadata(run_state, "run_state")
    if not isinstance(rngs, dict) or any(not isinstance(rng, Random) for rng in rngs.values()):
        raise ValueError("rngs must map names to random.Random instances")
    named_states = {name: rng.getstate() for name, rng in rngs.items()}
    _validate_rngs(named_states)
    dtype = _dtype_name(learner.online)
    online, target = _cpu_copy(learner.online.state_dict()), _cpu_copy(learner.target.state_dict())
    _validate_weights(online, dtype, architecture_version)
    _validate_weights(target, dtype, architecture_version)
    optimizer = _cpu_copy(learner.optimizer.state_dict())
    _validate_optimizer(optimizer, learner, optimizer_steps=learner.optimizer_steps)
    payload = {
        "format_version": CHECKPOINT_VERSION, "kind": "recovery",
        "architecture_version": architecture_version, "model_dtype": dtype,
        "config": learner.config.to_dict(), "online": online, "target": target,
        "optimizer": optimizer, "optimizer_steps": learner.optimizer_steps,
        "buffers": buffers.state_dict(), "run_state": run_state, "rngs": named_states,
        "global_rng": {"python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
                       "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else []},
    }
    _atomic_save(path, payload)


@restore_cpu_threads_on_error
def load_recovery(path: str | Path, *, device: str | None = None,
                  cpu_threads: int | None = None) -> RecoveryState:
    """Validate and restore recovery, then restore global RNG states on success.

    Override device with cpu/cuda/auto for portability. Same-device restoration
    preserves the next samples and optimizer update. CPU/CUDA numerical kernels
    can differ across devices. CUDA RNGs are restored for available matching
    device indices; extra saved devices are ignored on a smaller/CPU-only host.
    Optimizer load_state_dict moves AdamW moments to their parameter devices;
    its ordinary noncapturable step counters correctly remain on CPU.
    Apply the saved PyTorch CPU thread count before model construction, or an
    explicit override. The effective count is retained in the returned config.
    """
    payload = _load(path, "recovery", _RECOVERY_FIELDS)
    config = TrainingConfig.from_dict(payload["config"])
    if device is not None:
        config = replace(config, device=device)
    if cpu_threads is not None:
        config = replace(config, cpu_threads=cpu_threads)
    configure_cpu_threads(config.cpu_threads)
    _check_replay_snapshot_config(payload["buffers"], config)
    if type(payload["optimizer_steps"]) is not int or payload["optimizer_steps"] < 0:
        raise ValueError("invalid checkpoint optimizer_steps")
    run_state = _metadata(payload["run_state"], "run_state")
    rngs = _validate_rngs(payload["rngs"])
    _validate_global_rng(payload["global_rng"])
    online = _model(payload["online"], payload["model_dtype"], payload["architecture_version"])
    _validate_weights(payload["target"], payload["model_dtype"], payload["architecture_version"])
    learner = DQNLearner(config, online=online)
    _validate_optimizer(payload["optimizer"], learner, optimizer_steps=payload["optimizer_steps"])
    learner.target.load_state_dict(payload["target"], strict=True)
    learner.optimizer.load_state_dict(payload["optimizer"])
    learner.optimizer_steps = payload["optimizer_steps"]
    buffers = ReplayBuffers.from_state_dict(payload["buffers"])
    _check_buffers_config(buffers, config)
    global_rng = payload["global_rng"]
    random.setstate(global_rng["python"])
    torch.set_rng_state(global_rng["torch_cpu"])
    if torch.cuda.is_available():
        for index, state in enumerate(global_rng["torch_cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(state, index)
    return RecoveryState(learner, buffers, run_state, rngs)


def save_best(
    path: str | Path, online: QNetwork, *, config: TrainingConfig, metadata: dict[str, object],
) -> None:
    """Atomically save inference weights after the caller establishes improvement.

    Store configuration/format metadata and primitive evaluation details. This
    intentionally excludes the target, optimizer, replay and recovery counters.
    Saving does not change the supplied model's mode, weights or gradients.
    """
    if not isinstance(online, QNetwork) or not isinstance(config, TrainingConfig):
        raise ValueError("save_best requires a QNetwork and TrainingConfig")
    online.validate_architecture()
    if online.architecture_version != config.architecture_version:
        raise ValueError("checkpoint network must match the configured architecture_version")
    dtype = _dtype_name(online)
    state = _cpu_copy(online.state_dict())
    _validate_weights(state, dtype, online.architecture_version)
    payload = {"format_version": CHECKPOINT_VERSION, "kind": "best",
               "architecture_version": online.architecture_version, "model_dtype": dtype,
               "config": config.to_dict(), "online": state, "metadata": _metadata(metadata, "metadata")}
    _atomic_save(path, payload)


def load_best(path: str | Path, *, device: str = "cpu") -> BestCheckpoint:
    """Load a minimal inference checkpoint as a frozen, evaluation-mode QNetwork."""
    payload = _load(path, "best", _BEST_FIELDS)
    config = replace(TrainingConfig.from_dict(payload["config"]), device=device)
    metadata = _metadata(payload["metadata"], "metadata")
    model = _model(payload["online"], payload["model_dtype"], payload["architecture_version"])
    model.to(_resolve_device(config.device)).eval().requires_grad_(False)
    return BestCheckpoint(model, config, metadata)
