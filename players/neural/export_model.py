"""Export a training checkpoint as a permanent, frozen neural opponent.

Run from the project root with python -m players.neural.export_model. The source
is read-only, and this command neither resumes training nor adds the exported
player to a game/training pool. See docs/neural_player_exports.md for examples.
"""

import argparse
from pathlib import Path

from .artifacts import DEFAULT_WEIGHTS_DIR, save_neural_artifact


def _json_metadata(value):
    """Convert checkpoint primitive metadata into a JSON-compatible description.

    Checkpoints also allow tuples and bytes. Tuples become lists; bytes are
    recorded explicitly as hexadecimal, rather than dropping source details.
    The checkpoint reader has already checked that numbers are finite.
    """
    if isinstance(value, bytes):
        return {"encoding": "hex", "bytes": value.hex()}
    if isinstance(value, (tuple, list)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_metadata(item) for key, item in value.items()}
    return value


def export_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_root: str | Path = DEFAULT_WEIGHTS_DIR,
    name: str | None = None,
) -> Path:
    """Save the checkpoint's online network and return its permanent directory.

    Accept best.pt or latest.pt by their contents, not by their filename. Use
    the architecture recorded in that checkpoint (version 1 or version 2),
    the current encoder, action map and masked greedy selection with lowest-ID
    ties. Save the source path, file hash, training settings and available run
    metadata as provenance. Equal weights and inference conventions reuse the
    same identity regardless of source filename, label or training settings;
    an existing entry's original files and provenance remain unchanged.

    Do not load recovery state, restore RNGs or change the source checkpoint.
    The recovery file's serialized replay bytes are read but never turned into
    live replay buffers. No archive or source-run directory is required.
    """
    # Training checkpoint readers are needed only for export. Playing an
    # already exported opponent has no dependency on training configuration.
    from training.checkpoints import read_inference_checkpoint

    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("name must be a nonempty string or None")
    source = Path(checkpoint_path).resolve()
    checkpoint = read_inference_checkpoint(source)
    provenance = {
        "source_checkpoint": str(source),
        "source_sha256": checkpoint.source_sha256,
        "checkpoint_kind": checkpoint.kind,
        "training_config": checkpoint.config.to_dict(),
        "metadata": _json_metadata(checkpoint.metadata),
    }
    return save_neural_artifact(
        checkpoint.online, root=output_root, name=name, provenance=provenance,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse an export request, print the saved identity/path, and return zero."""
    parser = argparse.ArgumentParser(description=(
        "Save the online network from best.pt or latest.pt as a permanent neural opponent."
    ))
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Source best or recovery checkpoint (a .pt file, not a run folder).")
    parser.add_argument("--name", help="Readable label; does not change the model's permanent identity.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_WEIGHTS_DIR,
                        help="Destination library (default: project players/neural/weights).")
    arguments = parser.parse_args(argv)
    try:
        path = export_checkpoint(arguments.checkpoint, output_root=arguments.output_root,
                                 name=arguments.name)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    # Loading the saved label matters when a duplicate export reused an earlier
    # entry whose original name differs from this command's requested name.
    import json

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    print(f"Neural player: {manifest['name']}")
    print(f"Model ID: {manifest['model_id']}")
    print(f"Saved entry (created or reused): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
