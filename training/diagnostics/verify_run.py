"""Read-only consistency checks for a finished diagnostic update run."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_run(directory: Path) -> dict:
    """Check every completed game's callback records and immutable input hashes."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    games = [json.loads(line) for line in (directory / "games.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    budget = manifest["game_budget"]
    source_steps = manifest["source_optimizer_steps"]
    expected_batch = manifest["config"]["demo_batch_size"] + manifest["config"]["non_demo_batch_size"]
    counts = defaultdict(int)
    losses = defaultdict(float)
    last_steps = {}
    errors = []
    update_count = 0
    previous_game = 1
    trace_digest = hashlib.sha256()
    with (directory / "updates.jsonl").open("rb") as stream:
        for line in stream:
            trace_digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            update_count += 1
            game = row["game"]
            if not 1 <= game <= budget or game < previous_game:
                errors.append({"local_update": update_count, "problem": "invalid or nonmonotonic game", "game": game})
            previous_game = game
            expected_global = source_steps + update_count
            if (row["local_update"] != update_count or row["step"] != expected_global
                    or row["step_before"] != expected_global - 1):
                errors.append({"local_update": update_count, "problem": "nonconsecutive optimizer counters"})
            if row["batch_size"] != expected_batch:
                errors.append({"local_update": update_count, "problem": "wrong batch size"})
            if not math.isfinite(row["loss"]):
                errors.append({"local_update": update_count, "problem": "nonfinite loss"})
            counts[game] += 1
            losses[game] += row["loss"]
            last_steps[game] = row["step"]
    game_errors = []
    for game in games:
        number = game["game"]
        actual_count = counts[number]
        if game["updates"] != actual_count or game["transitions"] != actual_count:
            game_errors.append({"game": number, "problem": "transition/update/callback counts differ"})
        mean_loss = losses[number] / actual_count if actual_count else None
        if game["mean_loss"] != mean_loss:
            game_errors.append({"game": number, "problem": "game mean loss differs from actual callback losses"})
        if actual_count and game["optimizer_steps"] != last_steps[number]:
            game_errors.append({"game": number, "problem": "end-game global count differs from callback trace"})
    hashes = {}
    for label, path_key, hash_key in (
        ("source_checkpoint", "source_checkpoint", "source_sha256"),
        ("probe_bank", "probe_bank", "probe_bank_sha256"),
    ):
        actual = _sha256(Path(manifest[path_key]))
        hashes[label] = {
            "path": manifest[path_key], "expected_sha256": manifest[hash_key],
            "actual_sha256": actual, "unchanged": actual == manifest[hash_key],
        }
    checks = {
        "status_complete": summary["status"] == "complete",
        "games_exactly_1_through_budget_once": [game["game"] for game in games] == list(range(1, budget + 1)),
        "game_count_matches_summary": len(games) == summary["completed_games"] == budget,
        "callback_count_matches_summary": update_count == summary["optimizer_updates"],
        "callbacks_equal_total_transitions": update_count == sum(game["transitions"] for game in games),
        "callbacks_equal_total_game_updates": update_count == sum(game["updates"] for game in games),
        "global_update_delta_matches_callbacks": summary["final_optimizer_steps"] - source_steps == update_count,
        "every_callback_has_contiguous_counters_valid_game_batch_and_finite_loss": not errors,
        "every_game_mean_loss_and_counts_match_callbacks_exactly": not game_errors,
        "source_checkpoint_unchanged": hashes["source_checkpoint"]["unchanged"],
        "probe_bank_unchanged": hashes["probe_bank"]["unchanged"],
    }
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_dir": str(directory.resolve()), "all_checks_passed": all(checks.values()),
        "checks": checks, "game_budget": budget, "game_records": len(games),
        "callback_update_records": update_count,
        "total_transitions": sum(game["transitions"] for game in games),
        "total_game_updates": sum(game["updates"] for game in games),
        "source_optimizer_steps": source_steps,
        "final_optimizer_steps": summary["final_optimizer_steps"],
        "expected_batch_size": expected_batch,
        "updates_jsonl_sha256": trace_digest.hexdigest(),
        "games_jsonl_sha256": _sha256(directory / "games.jsonl"),
        "immutable_input_hashes": hashes,
        "callback_errors": errors, "game_errors": game_errors,
        "scope": "This verifies saved trace consistency and unchanged source/probe files. It does not independently replay simulator actions or recompute network gradients.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify_run(args.diagnostic_dir)
    path = args.diagnostic_dir / "verification.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({key: result[key] for key in (
        "all_checks_passed", "game_records", "callback_update_records", "checks",
    )}, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
