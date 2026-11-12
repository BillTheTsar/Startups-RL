"""Permanent, self-describing neural opponents, independent of training recovery.

Each directory contains manifest.json and weights.pt. Its neural_<SHA256> name
identifies the exact weights and inference rules, rather than a training run or
a mutable best.pt path. Saving the same policy again reuses the original entry
without rewriting its name, creation time or provenance. Loading checks both
compatibility and integrity before creating a frozen model.

Both declared QNetwork architectures, the current encoder, action codec and
legal greedy selection are supported. Version 1 retains its original identity;
version 2 includes hidden LayerNorm settings and weights in its identity.
Import this module explicitly when PyTorch is needed; normal player
imports remain framework-independent. No optimizer, replay or training archive
is read or created here.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile

import torch
from torch import Tensor

from action_codec import ActionCodec
from simulator import PlayerId

from .base import NeuralPlayer
from .encoder import FlatObservationEncoder
from .q_network import QNetwork


DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
ARTIFACT_VERSION = 1
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32, "float64": torch.float64}
_INFERENCE_FIELDS = ("architecture", "model_dtype", "observation_encoding", "action_codec", "selection")
_MANIFEST_FIELDS = {"format_version", "kind", "model_id", "name", "created_utc",
                    *_INFERENCE_FIELDS, "weights_file", "weights_sha256", "provenance"}


@dataclass(frozen=True)
class NeuralArtifact:
    """A validated saved policy and its frozen, evaluation-mode network.

    create_player gives each seat its own controller while sharing these frozen
    weights. Python still permits explicitly modifying a tensor; callers must
    keep this network out of optimizers and use a separate copy for training.
    """

    model_id: str
    name: str
    path: Path
    manifest: dict[str, object]
    online: QNetwork

    def create_player(
        self, player_id: PlayerId, *, seed: int | None = None, name: str | None = None,
    ) -> NeuralPlayer:
        """Create a fresh greedy controller, labeled with the permanent identity.

        Illegal actions are excluded and equal Q-values choose the lowest legal
        action ID. The optional seed belongs only to the controller; greedy
        choices do not consume it. No observation history is shared across seats.
        """
        return NeuralPlayer(
            player_id, self.online.predict, FlatObservationEncoder(),
            name=self.name if name is None else name, selection_mode="greedy",
            epsilon=0.0, seed=seed, checkpoint_id=self.model_id,
        )


def _json_bytes(value) -> bytes:
    """Use one deterministic JSON representation, preserving numeric types."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _json_copy(value):
    """Reject non-JSON metadata, including non-string keys and nonfinite values."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        _json_bytes(value)
        return value
    if type(value) in (list, tuple):
        return [_json_copy(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _json_copy(item) for key, item in value.items()}
    raise ValueError("artifact metadata must contain only finite JSON values and string-key dictionaries")


def _inference_metadata(dtype: str, architecture_version: int = 1) -> dict[str, object]:
    actions = []
    for action_id in range(ActionCodec.ACTION_COUNT):
        action = ActionCodec.decode(action_id)
        actions.append({"id": action_id, "kind": action.kind,
                        "company_id": action.company_id, "market_index": action.market_index})
    return {
        "architecture": QNetwork.architecture_metadata(architecture_version),
        "model_dtype": dtype,
        "observation_encoding": FlatObservationEncoder.metadata(),
        "action_codec": {"version": ActionCodec.VERSION, "action_count": ActionCodec.ACTION_COUNT,
                         "actions": actions},
        "selection": {"mode": "greedy", "legal_actions_only": True, "tie_break": "lowest_action_id"},
    }


def _validate_weights(state, dtype: str, architecture_version: int = 1) -> None:
    if type(dtype) is not str or dtype not in _DTYPES:
        raise ValueError("unsupported artifact model dtype")
    shapes = QNetwork.parameter_shapes(architecture_version)
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("artifact weights do not match the supported architecture")
    for name, shape in shapes.items():
        value = state[name]
        if not isinstance(value, Tensor) or value.layout != torch.strided or (
            value.device.type != "cpu" or value.dtype != _DTYPES[dtype] or tuple(value.shape) != shape
        ) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid artifact weight tensor: {name}")


def _identity(inference: dict, state: dict[str, Tensor]) -> str:
    """Hash semantics and sorted tensors using canonical little-endian bytes."""
    digest = hashlib.sha256()
    digest.update(b"startups.neural_player.identity.v1\0")
    descriptor = _json_bytes(inference)
    digest.update(struct.pack("<Q", len(descriptor)))
    digest.update(descriptor)
    for name, tensor in sorted(state.items()):
        name_bytes = name.encode("utf-8")
        digest.update(struct.pack("<Q", len(name_bytes)))
        digest.update(name_bytes)
        byte_view = tensor.detach().contiguous().view(torch.uint8)
        if sys.byteorder != "little":
            byte_view = byte_view.reshape(-1, tensor.element_size()).flip(1)
        data = byte_view.numpy().tobytes()
        digest.update(struct.pack("<Q", len(data)))
        digest.update(data)
    return "neural_" + digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(requested: str) -> torch.device:
    if type(requested) is not str or re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", requested) is None:
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
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


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest field: {key}")
        result[key] = value
    return result


def _read(path: Path) -> tuple[dict, dict[str, Tensor]]:
    """Validate the manifest, stored file hash and semantic policy identity."""
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"),
                              object_pairs_hook=_unique_json_object)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("artifact manifest is not readable JSON") from error
    if type(manifest) is not dict or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("artifact manifest fields do not match this format")
    _json_copy(manifest)
    if type(manifest["format_version"]) is not int or manifest["format_version"] != ARTIFACT_VERSION:
        raise ValueError("unsupported artifact format version")
    if manifest["kind"] != "startups.neural_player":
        raise ValueError("expected a neural player artifact")
    model_id = manifest["model_id"]
    if type(model_id) is not str or re.fullmatch(r"neural_[0-9a-f]{64}", model_id) is None:
        raise ValueError("invalid artifact model identity")
    if path.name != model_id:
        raise ValueError("artifact directory name must equal its model identity")
    if type(manifest["name"]) is not str or not manifest["name"].strip():
        raise ValueError("artifact name must be a nonempty string")
    if type(manifest["created_utc"]) is not str:
        raise ValueError("artifact creation time must be a UTC timestamp")
    try:
        created = datetime.fromisoformat(manifest["created_utc"])
    except ValueError as error:
        raise ValueError("invalid artifact creation time") from error
    if created.utcoffset() != timezone.utc.utcoffset(created):
        raise ValueError("artifact creation time must be in UTC")
    if type(manifest["provenance"]) is not dict:
        raise ValueError("artifact provenance must be a JSON dictionary")
    dtype = manifest["model_dtype"]
    if type(dtype) is not str or dtype not in _DTYPES:
        raise ValueError("unsupported artifact model dtype")
    inference = {name: manifest[name] for name in _INFERENCE_FIELDS}
    architecture = manifest["architecture"]
    if type(architecture) is not dict:
        raise ValueError("unsupported artifact architecture")
    architecture_version = architecture.get("version")
    expected = _inference_metadata(dtype, architecture_version)
    for name in _INFERENCE_FIELDS:
        if _json_bytes(inference[name]) != _json_bytes(expected[name]):
            raise ValueError(f"unsupported artifact {name}")
    if manifest["weights_file"] != "weights.pt":
        raise ValueError("artifact weights_file must be weights.pt")
    checksum = manifest["weights_sha256"]
    if type(checksum) is not str or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError("invalid artifact weights checksum")
    weights_path = path / "weights.pt"
    if _file_hash(weights_path) != checksum:
        raise ValueError("artifact weights file checksum does not match")
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except (FileNotFoundError, PermissionError):
        raise
    except Exception as error:
        raise ValueError("artifact weights are not a readable weights-only file") from error
    _validate_weights(state, dtype, architecture_version)
    if _identity(inference, state) != model_id:
        raise ValueError("artifact model identity does not match its weights and inference rules")
    return manifest, state


def _policy_snapshot(online: QNetwork) -> tuple[dict, dict[str, Tensor]]:
    """Copy and validate the exact inference content used for policy identity."""
    if type(online) is not QNetwork:
        raise ValueError("artifacts require an unmodified supported QNetwork architecture")
    online.validate_architecture()
    dtypes = {parameter.dtype for parameter in online.parameters()}
    dtype = next((label for label, value in _DTYPES.items() if dtypes == {value}), None)
    if dtype is None:
        raise ValueError("artifact models must use one supported floating-point dtype")
    state = {key: value.detach().cpu().clone() for key, value in online.state_dict().items()}
    _validate_weights(state, dtype, online.architecture_version)
    return _inference_metadata(dtype, online.architecture_version), state


def neural_policy_id(online: QNetwork) -> str:
    """Return the same permanent identity as export, without reading/writing files.

    Identity covers weights and supported inference rules. Display names and
    provenance do not affect it. Preserve the network's mode, gradients and RNG.
    """
    inference, state = _policy_snapshot(online)
    return _identity(inference, state)


def save_neural_artifact(
    online: QNetwork, *, root: str | Path = DEFAULT_WEIGHTS_DIR,
    name: str | None = None, provenance: dict | None = None,
) -> Path:
    """Publish a permanent inference-only copy and return its directory.

    Preserve the supplied network's weights, gradients, mode and RNG streams.
    Stage both files in a temporary sibling directory before publication. Never
    replace an existing model entry: identical content reuses it, while corrupt
    or incompatible content raises an error. Name and provenance describe the
    first export and do not affect identity or change on repeated exports.
    """
    inference, state = _policy_snapshot(online)
    if name is not None and (type(name) is not str or not name.strip()):
        raise ValueError("name must be a nonempty string or None")
    if provenance is not None and type(provenance) is not dict:
        raise ValueError("provenance must be a JSON dictionary or None")
    provenance = _json_copy({} if provenance is None else provenance)
    model_id = _identity(inference, state)
    root = Path(root).resolve()
    path = root / model_id
    if path.exists():
        _read(path)
        return path
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=root))
    try:
        weights_path = temporary / "weights.pt"
        with weights_path.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "format_version": ARTIFACT_VERSION, "kind": "startups.neural_player",
            "model_id": model_id, "name": name if name is not None else model_id,
            "created_utc": datetime.now(timezone.utc).isoformat(), **inference,
            "weights_file": "weights.pt", "weights_sha256": _file_hash(weights_path),
            "provenance": provenance,
        }
        with (temporary / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A fully populated destination cannot be replaced by os.rename on
            # Windows or POSIX. A concurrent identical export is safe to reuse.
            os.rename(temporary, path)
        except OSError:
            if not path.exists():
                raise
            _read(path)
        return path
    finally:
        if temporary.exists():
            # Delete only the exact temporary sibling created for this export.
            if temporary.resolve().parent != root or not temporary.name.startswith(".export-"):
                raise RuntimeError("refusing to clean an unexpected artifact temporary path")
            shutil.rmtree(temporary)


def load_neural_artifact(path: str | Path, *, device: str = "cpu") -> NeuralArtifact:
    """Validate and load one permanent opponent without restoring training state.

    Return an independent network in evaluation mode with gradients disabled.
    Default to CPU for portable inference; cpu, auto, cuda and cuda:N are
    supported. Loading preserves the caller's initialization RNG stream.
    """
    destination = _device(device)
    path = Path(path).resolve()
    manifest, state = _read(path)
    with torch.random.fork_rng(devices=[]):
        network = QNetwork(architecture_version=manifest["architecture"]["version"]).to(
            dtype=_DTYPES[manifest["model_dtype"]])
    network.load_state_dict(state, strict=True)
    network.to(destination).eval().requires_grad_(False)
    return NeuralArtifact(manifest["model_id"], manifest["name"], path, manifest, network)
