"""Human-readable and machine-readable exports derived entirely from an archive."""

from collections.abc import Sequence
import csv
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path

from evaluation import EvaluationResult, Statistics
from evaluation_storage import SavedRun, atomic_write_text, load_run


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(row[index]) for row in (headers, *rows)) for index in range(len(headers))]
    def line(row):
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))
    return "\n".join((line(headers), "-+-".join("-" * width for width in widths), *(line(row) for row in rows)))


def _number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.3f}" if value else "0.000"


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def _stat_row(stats: Statistics) -> tuple[str, ...]:
    return (
        str(stats.appearances), str(stats.capped_games), _number(stats.total_points), _number(stats.mean_points), str(stats.outright_wins),
        _percent(stats.outright_win_rate), *map(_percent, stats.placement_distribution),
    )


def format_report(result: EvaluationResult) -> str:
    """Render only the requested matrix; unavailable rates remain explicitly N/A."""
    labels = {model.model_id: f"M{index}" for index, model in enumerate(result.models, start=1)}
    lines = [f"Completed games: {result.completed_games} / {result.manifest['num_games']}",
             f"Ended at action cap: {result.capped_games}", "Model key:"]
    lines.extend(f"  {labels[model.model_id]} = {model.model_id}" for model in result.models)
    headers = ("Model", "Games", "Capped", "Total pts", "Mean pts", "Outright wins", "Win rate", "1st", "2nd", "3rd", "4th")
    ranked = sorted(result.models, key=lambda model: (
        model.overall.mean_points is None,
        -(model.overall.mean_points or 0), model.model_id,
    ))
    timeout_description = (
        "Timeouts score -1 each, with no placement credit; placement rates use all games."
        if result.manifest.get("evaluation_version", 1) >= 2
        else "This legacy run scored only normally finished games."
    )
    lines.extend((
        "", "Overall results (sorted by mean points)",
        _table(headers, [(labels[model.model_id], *_stat_row(model.overall)) for model in ranked]),
        "Win rate counts unique first-place finishes only; tied places split placement credit.",
        timeout_description,
        "", "Normalized head-to-head: row versus column; positive favors the row",
        "(wins - losses) / normally finished shared games, including ties; timeouts excluded.",
        "N/A = no normally finished shared game; diagonal = 0.",
        _table(("Model", *labels.values()), [
            (labels[model.model_id], *map(_number, row))
            for model, row in zip(result.models, result.head_to_head)
        ]),
    ))
    for position in range(4):
        lines.extend((
            "", f"Position {position + 1} relative to starter ({'first to act' if position == 0 else 'clockwise turn order'})",
            _table(headers, [(labels[model.model_id], *_stat_row(model.by_position[position])) for model in result.models]),
        ))
    return "\n".join(lines)


def _statistics_json(statistics: Statistics) -> dict:
    return {
        **asdict(statistics), "mean_points": statistics.mean_points,
        "outright_win_rate": statistics.outright_win_rate,
        "placement_distribution": statistics.placement_distribution,
        "capped_rate": statistics.capped_rate,
    }


def summary_data(saved: SavedRun) -> dict:
    """Include computed properties explicitly; dataclasses.asdict omits them."""
    result = saved.result
    return {
        "manifest": result.manifest,
        "run": saved.metadata,
        "completed_games": result.completed_games,
        "capped_games": result.capped_games,
        "models": [{"model_id": model.model_id, "overall": _statistics_json(model.overall),
                    "by_position": [{"position": position, **_statistics_json(statistics)}
                                    for position, statistics in enumerate(model.by_position, start=1)]}
                   for model in result.models],
        "head_to_head": {"model_ids": [model.model_id for model in result.models], "values": result.head_to_head},
    }


def _csv_text(headers, rows) -> str:
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    for row in rows:
        # N/A is distinct from a measured zero and is recognized by many readers.
        writer.writerow("N/A" if value is None else value for value in row)
    return stream.getvalue()


def _csv_statistics(statistics: Statistics) -> tuple:
    return (statistics.appearances, statistics.capped_games, statistics.capped_rate,
            statistics.total_points, statistics.mean_points,
            statistics.outright_wins, statistics.outright_win_rate,
            *statistics.placement_counts, *statistics.placement_distribution)


def format_saved_report(saved: SavedRun) -> str:
    lines = [f"Run: {saved.metadata['run_id']}", f"Status: {saved.metadata['status']}",
             f"Saved games: {saved.result.completed_games} / {saved.result.manifest['num_games']}"]
    if saved.metadata["status"] == "unfinished":
        lines.append("No final status was saved; this is the last committed snapshot, possibly from an active run.")
    if saved.metadata["error"]:
        lines.append("Recorded error: " + json.dumps(saved.metadata["error"], ensure_ascii=False))
    return "\n".join(lines) + "\n\n" + format_report(saved.result)


def export_reports(saved: SavedRun) -> None:
    """Replace derived files individually. A failed export never modifies the DB.

    All outputs come from one SQLite snapshot. Exported rates are numeric 0..1;
    placement counts can be fractional. Rebuild again if a file is locked or an
    interruption leaves reports from different snapshots. No replay is involved.
    """
    result = saved.result
    common = ("appearances", "capped_games", "capped_rate", "total_points", "mean_points", "outright_wins", "outright_win_rate",
              "first_credit", "second_credit", "third_credit", "fourth_credit",
              "first_rate", "second_rate", "third_rate", "fourth_rate")
    provenance = (saved.metadata["run_id"], saved.metadata["status"], result.completed_games,
                  result.manifest["num_games"])
    provenance_headers = ("run_id", "run_status", "saved_games", "requested_games")
    ranked = sorted(result.models, key=lambda model: (
        model.overall.mean_points is None, -(model.overall.mean_points or 0), model.model_id,
    ))
    outputs = {
        "manifest.json": json.dumps(result.manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "summary.json": json.dumps(summary_data(saved), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "overall.csv": _csv_text((*provenance_headers, "model_id", *common),
            [(*provenance, model.model_id, *_csv_statistics(model.overall)) for model in ranked]),
        "by_position.csv": _csv_text((*provenance_headers, "model_id", "position", *common),
            [(*provenance, model.model_id, position, *_csv_statistics(statistics))
             for model in result.models for position, statistics in enumerate(model.by_position, start=1)]),
        "head_to_head.csv": _csv_text(("model_id", *(model.model_id for model in result.models)),
            [(model.model_id, *row) for model, row in zip(result.models, result.head_to_head)]),
        "report.txt": format_saved_report(saved) + "\n",
    }
    for name, text in outputs.items():
        atomic_write_text(saved.directory / name, text)


def rebuild_reports(directory: Path | str) -> SavedRun:
    saved = load_run(directory)
    export_reports(saved)
    return saved


