"""One catalogue implementation for many explicitly selected opponent pools.

The library consists of the rule-based registry and exported neural directories.
A pool JSON chooses an ordered, unique subset by model ID. Loading a pool fixes
its membership and loads each selected neural model once; each game still gets
fresh player controllers. Nothing here samples seats or grows a training pool.

Importing this module, using rule-only pools, and listing available manifests do
not require PyTorch. Neural weights are checked and loaded only when selected.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from fractions import Fraction
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal, Protocol

from simulator import PlayerId
from .rule_based.registry import OpponentSpec, RULE_BASED_OPPONENTS

if TYPE_CHECKING:
    from .base import Player


DEFAULT_WEIGHTS_ROOT = Path(__file__).resolve().parent / "neural" / "weights"
_NEURAL_ID = re.compile(r"neural_[0-9a-f]{64}")
_INFERENCE_FIELDS = ("architecture", "model_dtype", "observation_encoding", "action_codec", "selection")


class PlayerFactory(Protocol):
    def __call__(self, player_id: PlayerId, *, seed: int | None) -> Player:
        """Construct a fresh controller, with independently seeded decision state."""
        ...


def _copy_config(config: Mapping[str, object]) -> dict:
    try:
        return json.loads(json.dumps(dict(config), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("config must contain finite JSON-compatible values") from error


def _weight_json(value):
    """Preserve exact Fraction settings without floating-point rounding."""
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {key: _weight_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_weight_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelSpec:
    """A policy's identity, factory and descriptive JSON settings, not a seat.

    Keep factories separate from JSON configuration: custom Python evaluation
    catalogues may supply any factory. Persistent training pools support the
    built-in rule families and validated permanent neural artifacts only.
    """

    model_id: str
    factory: PlayerFactory
    kind: Literal["rule_based", "neural"] = "rule_based"
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a nonempty string")
        if not callable(self.factory):
            raise ValueError("factory must be callable")
        if self.kind not in ("rule_based", "neural"):
            raise ValueError("evaluation models must be rule_based or neural")
        if not isinstance(self.config, Mapping):
            raise ValueError("config must be a JSON-compatible mapping")
        object.__setattr__(self, "config", _copy_config(self.config))

    def create(self, player_id: PlayerId, *, seed: int | None = None) -> Player:
        """Create a separate controller; a neural factory shares frozen weights."""
        return self.factory(player_id, seed=seed)

    @property
    def name(self) -> str:
        """Use the original player label while keeping model_id unambiguous."""
        owner = getattr(self.factory, "__self__", None)
        return getattr(owner, "name", self.config.get("name", self.model_id))

    @property
    def description(self) -> str:
        """Describe the existing rule strategy or frozen neural selection rule."""
        owner = getattr(self.factory, "__self__", None)
        if isinstance(owner, OpponentSpec):
            return owner.description
        if self.kind == "neural":
            return f"frozen neural opponent; highest legal Q-value; {self.model_id}"
        return "custom rule-based policy"


def as_model_spec(opponent: OpponentSpec | ModelSpec) -> ModelSpec:
    """Adapt a legacy registry entry without changing its factory or player label."""
    if isinstance(opponent, ModelSpec):
        return opponent
    if not isinstance(opponent, OpponentSpec):
        raise ValueError("opponents must contain OpponentSpec or ModelSpec entries")
    model_id = opponent.name
    config = {"class": f"{opponent.player_class.__module__}.{opponent.player_class.__qualname__}"}
    if opponent.weights is not None:
        model_id = opponent.name.removeprefix("Ranking (").removesuffix(")")
        config["weights"] = _weight_json(asdict(opponent.weights))
    if opponent.parameters is not None:
        config["parameters"] = asdict(opponent.parameters)
        config["chicken"] = opponent.chicken
    return ModelSpec(model_id, opponent.create, config=config)


def builtin_models() -> tuple[ModelSpec, ...]:
    """Return the existing rule-only catalogue in its original registry order."""
    return tuple(as_model_spec(opponent) for opponent in RULE_BASED_OPPONENTS)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _ids(values) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values or any(
        type(value) is not str or not value.strip() for value in values
    ):
        raise ValueError("model_ids must be a nonempty list of model IDs")
    if len(set(values)) != len(values):
        raise ValueError("pool model IDs must be unique")
    return tuple(values)


def _neural_config(artifact) -> dict:
    return {
        "artifact_path": str(artifact.path.resolve()), "name": artifact.name,
        "weights_sha256": artifact.manifest["weights_sha256"],
        **{key: artifact.manifest[key] for key in _INFERENCE_FIELDS},
    }


def load_neural_model(path: str | Path, *, device: str = "cpu") -> ModelSpec:
    """Load one permanent neural entry as a reusable pool member.

    Validate its inference metadata and weights, then share the frozen network
    between independently created controllers. This does not add the entry to
    any pool or change the catalogue's available-model inventory.
    """
    _validate_device(device)
    from .neural.artifacts import load_neural_artifact

    artifact = load_neural_artifact(path, device=device)
    return ModelSpec(artifact.model_id, artifact.create_player, "neural", _neural_config(artifact))


def _load_neural(path: Path, device: str) -> ModelSpec:
    """Retain the existing internal loader call signature."""
    return load_neural_model(path, device=device)


def load_pool(path: str | Path, *, device: str = "cpu") -> tuple[ModelSpec, ...]:
    """Resolve a selected JSON pool, loading only the neural IDs it requests.

    Require {"model_ids": [...]} with an optional "weights_root" string. An
    explicit relative root is relative to the JSON file; the default is the
    project's players/neural/weights directory. IDs retain their listed order.
    One model is enough for training/human play, which sample with replacement;
    standalone evaluation separately requires at least four distinct models.
    """
    _validate_device(device)
    path = Path(path).resolve()
    settings = _read_json(path)
    if type(settings) is not dict or set(settings) not in (
        {"model_ids"}, {"model_ids", "weights_root"},
    ):
        raise ValueError("pool JSON must contain model_ids and optionally weights_root")
    ids = _ids(settings["model_ids"])
    root = DEFAULT_WEIGHTS_ROOT
    if "weights_root" in settings:
        value = settings["weights_root"]
        if type(value) is not str or not value.strip():
            raise ValueError("weights_root must be a nonempty path string")
        root = Path(value)
        if not root.is_absolute():
            root = path.parent / root
    available = {model.model_id: model for model in builtin_models()}
    unknown = [model_id for model_id in ids if model_id not in available and _NEURAL_ID.fullmatch(model_id) is None]
    if unknown:
        raise ValueError("unknown model IDs: " + ", ".join(unknown))
    return tuple(available[model_id] if model_id in available else _load_neural(root / model_id, device)
                 for model_id in ids)


@dataclass(frozen=True)
class AvailableModel:
    """An inventory row; manifest listing does not validate/load neural tensors."""

    model_id: str
    name: str
    kind: Literal["rule_based", "neural"]
    artifact_path: Path | None = None


def list_available_models(*, weights_root: str | Path | None = None) -> tuple[AvailableModel, ...]:
    """List all rule IDs and neural manifest labels without importing PyTorch.

    Loading a selected pool performs the full neural integrity checks. Missing
    default libraries mean there are no exported opponents yet. Ignore temporary
    export directories and unrelated files, but report malformed model entries.
    """
    rows = [AvailableModel(model.model_id, model.name, model.kind) for model in builtin_models()]
    root = DEFAULT_WEIGHTS_ROOT if weights_root is None else Path(weights_root).resolve()
    if not root.exists():
        if weights_root is not None:
            raise FileNotFoundError(f"neural weights directory does not exist: {root}")
        return tuple(rows)
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or _NEURAL_ID.fullmatch(directory.name) is None:
            continue
        manifest = _read_json(directory / "manifest.json")
        if type(manifest) is not dict or manifest.get("model_id") != directory.name or (
            manifest.get("kind") != "startups.neural_player"
            or type(manifest.get("format_version")) is not int or manifest["format_version"] != 1
            or type(manifest.get("name")) is not str or not manifest["name"].strip()
        ):
            raise ValueError(f"invalid neural manifest identity/name: {directory}")
        rows.append(AvailableModel(directory.name, manifest["name"], "neural", directory.resolve()))
    return tuple(rows)


def _rule_config(opponent: OpponentSpec) -> dict:
    controller = opponent.create(1, seed=0)
    weights = getattr(controller, "weights", None)
    parameters = getattr(controller, "parameters", None)
    return _weight_json({
        "name": opponent.name,
        "class": f"{opponent.player_class.__module__}.{opponent.player_class.__qualname__}",
        "weights": asdict(weights) if weights is not None else None,
        "parameters": asdict(parameters) if parameters is not None else None,
        "chicken": getattr(controller, "chicken", False),
    })


def _rule_from_config(config) -> OpponentSpec:
    """Rebuild known families from exact recorded settings, never import a path."""
    from .rule_based.random_player import RandomPlayer
    from .rule_based.threshold_player import ThresholdParameters, ThresholdPlayer
    from .rule_based.weighted_ranking_player import RankingWeights, WeightedRankingPlayer

    supported = {f"{cls.__module__}.{cls.__qualname__}": cls
                 for cls in (RandomPlayer, ThresholdPlayer, WeightedRankingPlayer)}
    if type(config) is not dict or set(config) != {"name", "class", "weights", "parameters", "chicken"}:
        raise ValueError("saved rule configuration fields are invalid")
    if type(config["name"]) is not str or not config["name"].strip() or type(config["chicken"]) is not bool:
        raise ValueError("saved rule name/chicken flag is invalid")
    if type(config["class"]) is not str or config["class"] not in supported:
        raise ValueError("saved pools support only the built-in rule families")
    cls = supported[config["class"]]
    weights, parameters = None, None
    if cls is WeightedRankingPlayer:
        data = config["weights"]
        if type(data) is not dict or set(data) != {item.name for item in fields(RankingWeights)}:
            raise ValueError("saved ranking weights must include every weight")
        def decode(value):
            if isinstance(value, list):
                return tuple(decode(item) for item in value)
            if type(value) is dict and set(value) == {"numerator", "denominator"} and (
                type(value["numerator"]) is int and type(value["denominator"]) is int
                and value["denominator"] > 0
            ):
                return Fraction(value["numerator"], value["denominator"])
            if type(value) is int:
                return value
            raise ValueError("invalid saved ranking weight")
        weights = RankingWeights(**{key: decode(value) for key, value in data.items()})
    elif config["weights"] is not None:
        raise ValueError("only ranking players may have saved weights")
    if cls is ThresholdPlayer:
        data = config["parameters"]
        if type(data) is not dict or set(data) != {item.name for item in fields(ThresholdParameters)}:
            raise ValueError("saved thresholds must include every parameter")
        parameters = ThresholdParameters(**data)
    elif config["parameters"] is not None or config["chicken"]:
        raise ValueError("only threshold players may have thresholds/chicken mode")
    return OpponentSpec(config["name"], cls, weights, parameters, config["chicken"])


def snapshot_pool(opponents: Sequence[ModelSpec | OpponentSpec]) -> list[dict]:
    """Record an ordered fixed pool with exact settings and permanent model hashes.

    Records are independent of the selection JSON and the available library.
    Arbitrary custom evaluation factories cannot be safely reconstructed from
    JSON and are rejected here; existing custom evaluation itself remains valid.
    """
    models = tuple(as_model_spec(item) for item in opponents)
    _ids(tuple(model.model_id for model in models))
    records = []
    for model in models:
        owner = getattr(model.factory, "__self__", None)
        if model.kind == "rule_based" and isinstance(owner, OpponentSpec):
            config = _rule_config(owner)
            _rule_from_config(config)
        elif model.kind == "neural":
            from .neural.artifacts import NeuralArtifact
            if not isinstance(owner, NeuralArtifact) or model.model_id != owner.model_id:
                raise ValueError("saved neural pools require permanent artifact factories")
            config = _neural_config(owner)
            if _copy_config(model.config) != _copy_config(config):
                raise ValueError("neural specification metadata differs from its loaded artifact")
        else:
            raise ValueError("saved pools require built-in rule or permanent neural factories")
        records.append({"model_id": model.model_id, "kind": model.kind, "config": _copy_config(config)})
    return records


def _validate_device(device: str) -> None:
    if type(device) is not str or re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", device) is None:
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")


def restore_pool(records: Sequence[dict], *, device: str = "cpu") -> tuple[ModelSpec, ...]:
    """Restore only recorded members/settings; validate all saved neural hashes.

    The original pool JSON and source training checkpoints are unnecessary.
    Neural artifact folders at their recorded absolute paths must still exist.
    Files added to the library after the checkpoint never enter the restored pool.
    """
    _validate_device(device)
    if not isinstance(records, (list, tuple)) or not records or any(
        type(record) is not dict or set(record) != {"model_id", "kind", "config"} for record in records
    ):
        raise ValueError("invalid saved opponent catalogue")
    _ids(tuple(record["model_id"] for record in records))
    restored = []
    for record in records:
        config = _copy_config(record["config"]) if isinstance(record["config"], Mapping) else None
        if record["kind"] == "rule_based":
            source = _rule_from_config(config)
            spec = ModelSpec(record["model_id"], source.create, config=config)
        elif record["kind"] == "neural":
            if config is None or type(config.get("artifact_path")) is not str or not Path(config["artifact_path"]).is_absolute():
                raise ValueError("saved neural artifact_path must be absolute")
            spec = _load_neural(Path(config["artifact_path"]), device)
            if spec.model_id != record["model_id"] or _copy_config(spec.config) != config:
                raise ValueError("saved neural opponent metadata/hash differs from its artifact")
        else:
            raise ValueError("invalid saved opponent kind")
        restored.append(spec)
    result = tuple(restored)
    # Canonical comparison also rejects noncanonical Fraction encodings.
    if json.dumps(snapshot_pool(result), sort_keys=True) != json.dumps(records, sort_keys=True):
        raise ValueError("saved opponent catalogue is not canonical")
    return result


def main(argv=None) -> int:
    """List available library entries or explicitly selected pool entries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true", required=True)
    parser.add_argument("--weights-root", type=Path, help="library to inspect; default: project neural weights directory")
    parser.add_argument("--catalog-file", type=Path, help="list this selected pool instead of the whole available library")
    args = parser.parse_args(argv)
    if args.catalog_file is not None and args.weights_root is not None:
        parser.error("--catalog-file supplies its own weights_root")
    try:
        models = load_pool(args.catalog_file) if args.catalog_file is not None else list_available_models(weights_root=args.weights_root)
        for model in models:
            print(f"{model.model_id} [{model.kind}] {model.name}")
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
