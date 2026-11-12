"""Evaluate automated Startups players: python evaluate.py --help.

Archive outcomes automatically and export statistics. Rebuild exports from a
saved archive without running games. Existing runs are never resumed/extended.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import asdict
from importlib import import_module
import json
from secrets import randbits
import sys

from evaluation import (
    DEFAULT_MAX_ACTIONS, EvaluationGameError, GameRecord,
    builtin_models, evaluate, validate_models,
)
from evaluation_reports import format_report, format_saved_report, rebuild_reports
from evaluation_storage import ArchiveError, DEFAULT_OUTPUT_ROOT, RunArchive


def load_pool(path, *, device="cpu"):
    """Load selected models only for games/listing, never for report rebuilding."""
    from players.catalogue import load_pool as load_selected_pool

    return load_selected_pool(path, device=device)


def _load_catalog(reference: str) -> tuple[ModelSpec, ...]:
    """Load a user-supplied sequence of specs without embedding checkpoint loading."""
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--catalog must have the form module:attribute")
    return tuple(getattr(import_module(module_name), attribute))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate four distinct automated players per game.")
    parser.add_argument("--games", type=int, help="total number of games (required unless listing models or rebuilding reports)")
    parser.add_argument("--models", nargs="+", metavar="ID", help="at least four distinct IDs; defaults to the whole catalog")
    parser.add_argument("--list-models", action="store_true", help="list available model IDs and exit")
    catalogs = parser.add_mutually_exclusive_group()
    catalogs.add_argument("--catalog", metavar="MODULE:ATTRIBUTE", help="custom sequence of ModelSpec entries, replacing the built-in catalog")
    catalogs.add_argument("--catalog-file", metavar="JSON", help="JSON file selecting rule-based and exported neural models")
    parser.add_argument("--opponent-device", help="neural model device for --catalog-file: cpu (default), auto, cuda, or cuda:<index>")
    parser.add_argument("--seed", type=int, help="master seed; generated and printed if omitted")
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS, help="per-game action limit (default: %(default)s)")
    parser.add_argument("--progress-every", type=int, default=100, help="report every N completed games to stderr; 0 disables (default: %(default)s)")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="parent folder for a new unique run (default: %(default)s)")
    parser.add_argument("--rebuild-report", metavar="RUN_FOLDER", help="rebuild exports from saved outcomes without playing games")
    args = parser.parse_args(argv)
    if args.rebuild_report:
        if (args.games is not None or args.models is not None or args.catalog or args.catalog_file
                or args.opponent_device is not None or args.list_models or args.seed is not None):
            parser.error("--rebuild-report cannot be combined with game, model, seed, or catalog options")
        try:
            saved = rebuild_reports(args.rebuild_report)
        except (ArchiveError, OSError) as error:
            print(f"Report rebuild failed: {error}", file=sys.stderr)
            return 1
        print(format_saved_report(saved))
        print(f"Exports rebuilt in: {saved.directory}")
        return 0
    try:
        from evaluation import ModelSpec

        if args.opponent_device is not None and args.catalog_file is None:
            raise ValueError("--opponent-device requires --catalog-file")
        if args.catalog_file:
            catalog = load_pool(args.catalog_file, device=args.opponent_device or "cpu")
        else:
            catalog = _load_catalog(args.catalog) if args.catalog else builtin_models()
        if any(not isinstance(spec, ModelSpec) for spec in catalog):
            raise ValueError("catalog entries must be ModelSpec instances")
        available = {spec.model_id: spec for spec in catalog}
        if len(available) != len(catalog):
            raise ValueError("catalog model IDs must be unique")
        if args.list_models:
            for model in catalog:
                print(f"{model.model_id} [{model.kind}]")
            return 0
        if args.games is None or args.games <= 0:
            raise ValueError("--games must be a positive integer")
        if args.max_actions <= 0 or args.progress_every < 0:
            raise ValueError("--max-actions must be positive and --progress-every must be nonnegative")
        selected_ids = tuple(available) if args.models is None else tuple(args.models)
        unknown = [model_id for model_id in selected_ids if model_id not in available]
        if unknown:
            raise ValueError(f"unknown model IDs: {', '.join(unknown)}; use --list-models")
        models = validate_models(tuple(available[model_id] for model_id in selected_ids))
    except (ValueError, ImportError, AttributeError, TypeError, OSError) as error:
        parser.error(str(error))

    seed = randbits(64) if args.seed is None else args.seed
    print(f"Run seed: {seed}", flush=True)
    archive = None

    def start(manifest: dict) -> None:
        nonlocal archive
        archive = RunArchive(manifest, args.output_root)
        print(f"Run folder: {archive.directory}", flush=True)

    def check_storage() -> None:
        if archive is not None:
            archive.check()

    def progress(record: GameRecord) -> None:
        archive.append(record)
        if args.progress_every and (record.plan.game_number % args.progress_every == 0 or record.plan.game_number == args.games):
            print(f"Completed {record.plan.game_number}/{args.games} games", file=sys.stderr, flush=True)

    status, details, exit_code = "completed", None, 0
    try:
        asyncio.run(evaluate(models, args.games, seed=seed, max_actions=args.max_actions,
                             on_game=progress, on_start=start, check_status=check_storage))
    except EvaluationGameError as error:
        print(str(error), file=sys.stderr)
        status, details, exit_code = "failed", {"message": str(error), "game_plan": asdict(error.plan)}, 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Evaluation interrupted.", file=sys.stderr)
        status, details, exit_code = "interrupted", {"message": "Evaluation interrupted"}, 130
    except Exception as error:
        print(f"Evaluation stopped: {error}", file=sys.stderr)
        status, details, exit_code = "failed", {"message": f"{type(error).__name__}: {error}"}, 1
    finally:
        if archive is not None:
            try:
                archive.finish(status, details)
            except ArchiveError as error:
                print(str(error), file=sys.stderr)
                exit_code = 1
    if archive is None:
        return exit_code or 1
    try:
        saved = rebuild_reports(archive.directory)
        print("Run manifest:\n" + json.dumps(saved.result.manifest, indent=2))
        print(format_saved_report(saved))
        print(f"Saved results: {archive.directory}")
    except (ArchiveError, OSError) as error:
        print(f"Could not export reports: {error}. Archive location: {archive.directory}", file=sys.stderr)
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
