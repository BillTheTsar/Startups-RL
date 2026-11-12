"""Compare a diagnostic game's recorded measurements with a historical run.

This module only reads training records. It never loads a network, writes to an
archive, or runs a game. Exact agreement concerns the recorded per-game fields;
it does not prove that every individual parameter or action is identical.

The JSONL reader takes a bounded snapshot and ignores an unfinished last line,
so this can run while the diagnostic process is appending its next record.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


GAME_FIELDS = (
    "epsilon", "reward", "capped", "actions", "transitions", "updates",
    "optimizer_steps", "mean_loss", "buffer_sizes",
)


def _read_json(path: Path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def read_game_snapshot(path: Path) -> tuple[list[dict], dict]:
    """Read only complete JSONL rows present when the file was opened."""
    with path.open("rb") as stream:
        size = path.stat().st_size
        data = stream.read(size)
    boundary = data.rfind(b"\n") + 1
    complete = data[:boundary]
    rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
    return rows, {
        "bytes_captured": len(data), "complete_bytes": len(complete),
        "incomplete_trailing_bytes": len(data) - len(complete),
        "complete_bytes_sha256": hashlib.sha256(complete).hexdigest(),
        "complete_rows": len(rows),
    }


def compare_games(original: list[dict], diagnostic: list[dict]) -> dict:
    """Use exact Python equality, including loss floats, for each saved field."""
    by_game = {row["game"]: row for row in original}
    if len(by_game) != len(original):
        raise ValueError("Historical metrics contain duplicate game numbers")
    differences = []
    mismatched_games = 0
    consecutive_prefix = 0
    counts = {field: 0 for field in GAME_FIELDS}
    expected = 1
    for row in diagnostic:
        number = row["game"]
        if number != expected:
            raise ValueError(f"Diagnostic game sequence expected {expected}, got {number}")
        expected += 1
        source = by_game.get(number)
        game_matches = True
        for field in GAME_FIELDS:
            if source is None or field not in source or field not in row or source[field] != row[field]:
                counts[field] += 1
                game_matches = False
                if len(differences) < 30:
                    differences.append({
                        "game": number, "field": field,
                        "historical": source.get(field) if source else None,
                        "diagnostic": row.get(field),
                    })
        if game_matches and consecutive_prefix == number - 1:
            consecutive_prefix = number
        if not game_matches:
            mismatched_games += 1
    return {
        "fields": list(GAME_FIELDS), "comparison": "exact equality; no tolerance",
        "compared_games": len(diagnostic), "matched_prefix_games": consecutive_prefix,
        "mismatched_games": mismatched_games,
        "mismatch_counts_by_field": counts,
        "first_differences_up_to_30": differences,
        "all_recorded_fields_equal": bool(diagnostic) and mismatched_games == 0,
    }


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_losses(metrics: dict, *, window: int = 1_000) -> dict:
    """Describe saved per-game mean MSE without treating it as a gradient log."""
    games = metrics["games"]
    usable = [row for row in games if isinstance(row.get("mean_loss"), (int, float))
              and math.isfinite(row["mean_loss"])]
    evaluations = metrics.get("evaluations", [])

    def compact(row):
        return {key: row.get(key) for key in (
            "game", "mean_loss", "epsilon", "reward", "capped", "updates",
            "optimizer_steps",
        )}

    def neighbors(game):
        before = [row for row in evaluations if row["game"] <= game]
        after = [row for row in evaluations if row["game"] > game]
        return [
            {key: row.get(key) for key in (
                "game", "mean_reward", "capped_games", "qualifies", "pool_size_after",
            )}
            for row in before[-2:] + after[:2]
        ]

    windows = []
    for start in range(1, max((row["game"] for row in games), default=0) + 1, window):
        rows = [row for row in usable if start <= row["game"] < start + window]
        if not rows:
            continue
        values = [row["mean_loss"] for row in rows]
        maximum = max(rows, key=lambda row: row["mean_loss"])
        windows.append({
            "first_game": start, "last_game": start + window - 1, "count": len(rows),
            "median": median(values), "p95": _quantile(values, .95),
            "mean": mean(values), "max": maximum["mean_loss"],
            "max_game": maximum["game"],
            "above_1_games": sum(value > 1 for value in values),
            "above_5_games": sum(value > 5 for value in values),
            "above_10_games": sum(value > 10 for value in values),
        })
    extremes = {}
    for threshold in (1, 5, 10):
        rows = [row for row in usable if row["mean_loss"] > threshold]
        first = compact(rows[0]) if rows else None
        if first:
            first["nearby_evaluations"] = neighbors(first["game"])
        extremes[f"strictly_above_{threshold}"] = {
            "count": len(rows), "first": first,
            "before_first_surviving_mixed_checkpoint_26800": sum(row["game"] < 26800 for row in rows),
        }
    top = []
    for row in sorted(usable, key=lambda row: row["mean_loss"], reverse=True)[:10]:
        entry = compact(row)
        entry["nearby_evaluations"] = neighbors(row["game"])
        top.append(entry)
    return {
        "run_id": metrics.get("run_id"), "completed_games": metrics.get("completed_games"),
        "usable_game_losses": len(usable), "missing_or_nonfinite_losses": len(games) - len(usable),
        "meaning": "Each value is the arithmetic mean of recorded batch MSEs during one training game, not one individual update.",
        "cautions": [
            "Thresholds 1, 5 and 10 are descriptive cutoffs, not validated failure criteria.",
            "A game mean hides within-game peaks and does not measure gradients or hidden-neuron activation.",
            "Windows weight each game equally, although games contain different numbers of updates.",
            "Loss alone cannot establish when hidden units became inactive or why Q-values changed.",
        ],
        "window_size_games": window, "windows": windows,
        "threshold_crossings": extremes, "top_10_game_mean_losses": top,
    }


def compare_history(metrics_path: Path, diagnostic_dir: Path, *, final: bool = False) -> tuple[dict, dict]:
    metrics, metrics_sha = _read_json(metrics_path)
    manifest, manifest_sha = _read_json(diagnostic_dir / "manifest.json")
    rows, snapshot = read_game_snapshot(diagnostic_dir / "games.jsonl")
    result = compare_games(metrics["games"], rows)
    budget = manifest["game_budget"]
    summary_path = diagnostic_dir / "summary.json"
    summary = _read_json(summary_path)[0] if summary_path.exists() else None
    if final and (len(rows) != budget or summary is None or summary.get("status") != "complete"):
        raise ValueError("--final requires the completed diagnostic summary and all budgeted game records")
    config_differences = {
        key: {"historical": metrics["config"].get(key), "diagnostic": manifest["config"].get(key)}
        for key in sorted(set(metrics["config"]) | set(manifest["config"]))
        if metrics["config"].get(key) != manifest["config"].get(key)
    }
    parent = metrics.get("parent", {})
    inherited_fields = {
        "source_run_id": "source_run_id", "completed_games": "source_completed_games",
        "optimizer_steps": "source_optimizer_steps", "rng_mode": "rng_mode",
    }
    parent_matches = {
        source_key: parent.get(source_key) == manifest.get(manifest_key)
        for source_key, manifest_key in inherited_fields.items()
    }
    parent_matches["checkpoint_path"] = Path(parent["checkpoint_path"]).resolve() == Path(manifest["source_checkpoint"]).resolve()
    evaluations = []
    for row in metrics.get("evaluations", []):
        if row["game"] <= budget:
            evaluations.append({
                "game": row["game"], "mean_reward": row["mean_reward"],
                "qualifies": row.get("qualifies"), "admission": row.get("admission"),
                "evaluated_pool_size": len(row.get("evaluated_opponents", [])),
                "evaluated_pool_matches_diagnostic": row.get("evaluated_opponents") == manifest["opponents"],
                "pool_size_after": row.get("pool_size_after"),
                "buffer_sizes": row.get("buffer_sizes"),
            })
    refreshes = [{key: row.get(key) for key in (
        "game", "before_buffer_sizes", "buffer_sizes", "archive_snapshot",
    )} for row in metrics.get("refreshes", []) if row["game"] <= budget]
    result.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "final" if final else "partial",
        "historical_metrics_path": str(metrics_path.resolve()), "historical_metrics_sha256": metrics_sha,
        "historical_run_id": metrics["run_id"], "diagnostic_dir": str(diagnostic_dir.resolve()),
        "diagnostic_manifest_sha256": manifest_sha, "captured_game_records": snapshot,
        "budget": budget, "diagnostic_summary_status": summary.get("status") if summary else None,
        "configuration_differences": config_differences, "parent_fields_match": parent_matches,
        "original_parent": parent, "original_evaluations_within_budget": evaluations,
        "original_admissions_within_budget": [row for row in metrics.get("admissions", []) if row["completed_games"] <= budget],
        "original_refreshes_at_or_before_budget": refreshes,
        "interpretation_limits": [
            "Exact agreement applies to the saved per-game measurements, including mean loss; historical per-update weights and action sequences were not saved for this comparison.",
            "The original evaluations within this window can be omitted without changing replay or opponents only if they promoted nothing and admitted nobody; the evidence above records that check.",
            "An original refresh after the final budgeted game changes subsequent replay, not the training updates inside that final game. The diagnostic deliberately stops before refreshing.",
            "This comparison only covers the measured prefix. It says nothing about the later collapse unless that period was actually rerun.",
        ],
    })
    losses = summarize_losses(metrics)
    losses["historical_metrics_path"] = str(metrics_path.resolve())
    losses["historical_metrics_sha256"] = metrics_sha
    return result, losses


def _write(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    comparison, losses = compare_history(args.metrics, args.diagnostic_dir, final=args.final)
    _write(args.diagnostic_dir / "history_comparison.json", comparison)
    _write(args.diagnostic_dir / "historical_loss_summary.json", losses)
    print(json.dumps({key: comparison[key] for key in (
        "status", "compared_games", "matched_prefix_games", "mismatched_games",
        "all_recorded_fields_equal", "configuration_differences", "parent_fields_match",
    )}, indent=2))


if __name__ == "__main__":
    main()
