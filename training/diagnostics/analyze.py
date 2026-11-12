"""Read diagnostic logs and make a small report, without importing training code.

Run this file directly with a Python installation containing NumPy and, for
figures, Matplotlib. It never loads checkpoints or changes a running experiment.
Large update logs are streamed into numeric columns; only a few extreme records
are retained. A growing log is read only through its size when reading begins,
and its last unfinished line is ignored. Run again after training for final data.
"""

from __future__ import annotations

import argparse
from array import array
import csv
from datetime import datetime, timezone
import heapq
import json
from pathlib import Path

import numpy as np


SCALARS = {
    "game": "game", "local_update": "local_update", "step": "step",
    "elapsed_seconds": "elapsed_seconds", "loss": "loss",
    "selected_q_mean": "selected_q.mean", "selected_q_abs_max": "selected_q.abs_max",
    "max_legal_q_max": "max_legal_q.max", "max_legal_q_p99": "max_legal_q.p99",
    "target_abs_max": "targets.abs_max", "td_abs_mean": "abs_td_error.mean",
    "td_abs_max": "abs_td_error.max", "gradient_l2": "gradient_l2",
    "gradient_abs_max": "gradient_abs_max", "update_l2": "update_l2",
    "relative_update_l2": "relative_update_l2", "terminal_count": "terminal_count",
    "terminal_td_mse": "terminal.td_mse", "nonterminal_td_mse": "nonterminal.td_mse",
    "gradient_nonfinite_count": "gradient_nonfinite_count",
}
for _layer in ("hidden_1", "hidden_2", "hidden_3", "output"):
    for _field in ("gradient_l2", "update_l2", "relative_update_l2"):
        SCALARS[f"{_layer}_{_field}"] = f"layers.{_layer}.{_field}"
for _layer in ("hidden_1", "hidden_2", "hidden_3"):
    for _field in ("active_neurons", "all_zero_row_fraction", "pre_min", "pre_max"):
        SCALARS[f"{_layer}_{_field}"] = f"batch_activations.{_layer}.{_field}"


def nested_number(record, path):
    """Read a numeric field, recording missing or explicitly nonfinite values as NaN."""
    value = record
    for part in path.split("."):
        if not isinstance(value, dict):
            return float("nan")
        value = value.get(part)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def complete_lines(path, metadata):
    """Yield complete JSON records from a bounded view of an append-only file."""
    size = path.stat().st_size if path.exists() else 0
    metadata.update(path=str(path), bytes_at_start=size, bytes_read=0, records=0,
                    unfinished_last_line=False)
    if not size:
        return
    with path.open("rb") as stream:
        while stream.tell() < size:
            line = stream.readline(size - stream.tell())
            if not line.endswith(b"\n"):
                metadata["unfinished_last_line"] = True
                break
            metadata["bytes_read"] = stream.tell()
            if line.strip():
                # A malformed completed record is an error, not silently discarded.
                record = json.loads(line)
                metadata["records"] += 1
                yield record


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result = {"count": int(values.size), "nonfinite_count": int(values.size - finite.size)}
    if finite.size:
        result.update(zip(("min", "median", "p95", "p99", "max"),
                          map(float, np.quantile(finite, [0, .5, .95, .99, 1]))))
        result["mean"] = float(finite.mean())
    return result


def read_updates(path):
    """Keep numeric columns and top-ten examples; do not retain every JSON object."""
    columns = {name: array("d") for name in SCALARS}
    top = {name: [] for name in ("td_abs_max", "gradient_l2", "update_l2", "relative_update_l2")}
    metadata = {}
    for record in complete_lines(path, metadata):
        values = {name: nested_number(record, key) for name, key in SCALARS.items()}
        for name, value in values.items():
            columns[name].append(value)
        for metric, heap in top.items():
            value = values[metric]
            if np.isfinite(value) and (len(heap) < 10 or value > heap[0][0]):
                item = (value, metadata["records"], values.copy())
                if len(heap) < 10:
                    heapq.heappush(heap, item)
                else:
                    heapq.heapreplace(heap, item)
    return ({key: np.asarray(value) for key, value in columns.items()},
            {key: [item[2] for item in sorted(heap, reverse=True)] for key, heap in top.items()},
            metadata)


def summarize_probe(record):
    result = {name: record.get(name) for name in
              ("game", "completed_games", "local_updates", "optimizer_steps", "elapsed_seconds", "reason")}
    for name in ("online", "target"):
        measurement = record.get(name)
        if measurement is None:
            result[name] = None
            continue
        result[name] = {
            "max_legal_q": measurement["max_legal_q"],
            "hidden_layers": [{key: layer[key] for key in
                               ("width", "ever_active_units", "all_zero_rows", "all_zero_fraction",
                                "activation_fraction")} for layer in measurement["hidden_layers"]],
            "last_layer_zero_vs_q": measurement["last_layer_zero_vs_q"],
        }
    return result


WINDOW_METRICS = ("loss", "td_abs_max", "gradient_l2", "update_l2", "relative_update_l2",
                  "selected_q_abs_max", "target_abs_max", "max_legal_q_max")


def window_summaries(columns, size=100):
    result = []
    for start in range(0, len(columns["step"]), size):
        end = min(start + size, len(columns["step"]))
        result.append({"first_local_update": int(columns["local_update"][start]),
                       "last_local_update": int(columns["local_update"][end - 1]),
                       "first_game": int(columns["game"][start]),
                       "last_game": int(columns["game"][end - 1]),
                       "count": end - start,
                       **{name: stats(columns[name][start:end]) for name in WINDOW_METRICS}})
    return result


def probe_intervals(probes, columns):
    """Associate measured changes with updates between probes, without causal claims."""
    result = []
    for before, after in zip(probes, probes[1:]):
        first, last = before["local_updates"], after["local_updates"]
        if last <= first:
            continue
        start, end = np.searchsorted(columns["local_update"], [first, last], side="right")
        b, a = before["online"], after["online"]
        if b is None or a is None:
            continue
        bh, ah = b["hidden_layers"][-1], a["hidden_layers"][-1]
        result.append({
            "first_local_update": first, "last_local_update": last,
            "first_game": before["game"], "last_game": after["game"],
            "covered_updates": int(end - start),
            "last_layer_active_before": bh["ever_active_units"],
            "last_layer_active_after": ah["ever_active_units"],
            "active_unit_change": ah["ever_active_units"] - bh["ever_active_units"],
            "zero_fraction_change": ah["all_zero_fraction"] - bh["all_zero_fraction"],
            "q_max_before": b["max_legal_q"]["max"], "q_max_after": a["max_legal_q"]["max"],
            "q_max_change": a["max_legal_q"]["max"] - b["max_legal_q"]["max"],
            "q_p99_change": a["max_legal_q"]["p99"] - b["max_legal_q"]["p99"],
            "updates": {name: stats(columns[name][start:end]) for name in WINDOW_METRICS},
        })
    return result


def target_changes(probes):
    """Compare target probes to earlier online probes, retaining the time ordering."""
    changes = []
    previous_target = None
    for index, current in enumerate(probes):
        if current["target"] is None:
            continue
        if previous_target is not None:
            between = [probe for probe in probes[:index]
                       if probe["local_updates"] >= previous_target["local_updates"]]
            earlier_online = between[-1] if between else previous_target
            changes.append({
                "previous_target_local_update": previous_target["local_updates"],
                "target_local_update": current["local_updates"], "game": current["game"],
                "last_earlier_online_local_update": earlier_online["local_updates"],
                "last_earlier_online_q": earlier_online["online"]["max_legal_q"],
                "current_online_q": current["online"]["max_legal_q"],
                "previous_target_q": previous_target["target"]["max_legal_q"],
                "current_target_q": current["target"]["max_legal_q"],
                "target_q_max_change": current["target"]["max_legal_q"]["max"] - previous_target["target"]["max_legal_q"]["max"],
                "largest_prior_online_q_max": max(probe["online"]["max_legal_q"]["max"] for probe in between),
            })
        previous_target = current
    return changes


def game_window_summary(columns, probes, first_game, last_game):
    """Compare equal game ranges; label summaries of repeated probe readings."""
    selected = (columns["game"] >= first_game) & (columns["game"] <= last_game)
    readings = [probe for probe in probes if first_game <= probe["game"] <= last_game]
    return {
        "first_game": first_game, "last_game": last_game,
        "updates": int(selected.sum()), "probes": len(readings),
        "update_metrics": {name: stats(columns[name][selected]) for name in WINDOW_METRICS},
        "online_probe_readings": {
            field: stats([probe["online"]["max_legal_q"][field] for probe in readings])
            for field in ("median", "p99", "max")
        },
        "target_probe_readings": {
            field: stats([probe["target"]["max_legal_q"][field] for probe in readings if probe["target"]])
            for field in ("median", "p99", "max")
        },
        "active_last_units": stats([probe["online"]["hidden_layers"][-1]["ever_active_units"] for probe in readings]),
        "zero_last_fraction": stats([probe["online"]["hidden_layers"][-1]["all_zero_fraction"] for probe in readings]),
    }


def activity_excursions(probes):
    """Find count drops and later count recovery; this does not match neuron identities."""
    if not probes:
        return {"largest_single_interval_drops": [], "largest_peak_to_later_trough": None}
    counts = np.array([probe["online"]["hidden_layers"][-1]["ever_active_units"] for probe in probes])

    def point(index):
        return {"game": probes[index]["game"], "local_updates": probes[index]["local_updates"],
                "active_units": int(counts[index])}

    def recovery(after, threshold):
        candidates = np.flatnonzero(counts[after + 1:] >= threshold)
        return point(after + 1 + int(candidates[0])) if len(candidates) else None

    drops = []
    for index in np.flatnonzero(np.diff(counts) < 0):
        drops.append({"before": point(index), "after": point(index + 1),
                      "units_lost": int(counts[index] - counts[index + 1]),
                      "count_recovered_at": recovery(index + 1, counts[index])})
    peak, worst = 0, None
    for index, count in enumerate(counts):
        if count > counts[peak]:
            peak = index
        loss = int(counts[peak] - count)
        if worst is None or loss > worst["units_lost"]:
            worst = {"peak": point(peak), "trough": point(index), "units_lost": loss,
                     "count_recovered_at": recovery(index, counts[peak])}
    return {"largest_single_interval_drops": sorted(drops, key=lambda row: row["units_lost"], reverse=True)[:10],
            "largest_peak_to_later_trough": worst,
            "meaning": "Recovery means the active-unit count returned; it does not prove the same neuron identities recovered."}


def summarize_timeline(path):
    """Keep related parent/mixed lineage snapshots separate from unrelated runs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    parent_id = "8060854903a6"
    mixed_id = "e9f8a65f5914"
    rows = []
    for model in data["models"]:
        selected = {}
        for source in model["sources"]:
            run_id = source.get("run_id", "") or ""
            if run_id.startswith(parent_id) or run_id.startswith(mixed_id):
                selected.setdefault((run_id, source.get("game")), source)
        for (run_id, game), source in selected.items():
            measurement = model["online"]
            rows.append({"run_id": run_id, "game": game,
                         "mixed_run_game": game if run_id.startswith(mixed_id) else game - 35800,
                         "optimizer_steps": source.get("optimizer_steps"),
                         "model_id": model["model_id"], "source": source,
                         "max_legal_q": measurement["max_legal_q"],
                         "hidden_layers": [{key: layer[key] for key in
                                            ("ever_active_units", "all_zero_rows", "all_zero_fraction",
                                             "activation_fraction")} for layer in measurement["hidden_layers"]],
                         "last_layer_zero_vs_q": measurement["last_layer_zero_vs_q"],
                         "largest_q_examples": measurement["largest_q_examples"][:3]})
    rows.sort(key=lambda row: row["optimizer_steps"])
    return {"snapshots": rows, "limitations": data["limitations"],
            "probe_bank": data["probe_bank"]}


def finite_json(value):
    """Use null for missing/nonfinite values so output remains strict JSON."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def save_csv(path, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def make_figures(figure_dir, timeline, probes, windows, completed_games, final):
    """Draw standalone figures; Matplotlib is optional for numeric analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    paths = []
    if timeline:
        rows = [row for row in timeline["snapshots"] if row["mixed_run_game"] >= 0]
        x = [row["mixed_run_game"] for row in rows]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
        axes[0].plot(x, [row["max_legal_q"]["max"] for row in rows], "o--", color="#b44535")
        axes[0].set(yscale="log", ylabel="Largest legal Q\n(log scale)")
        axes[0].axhline(1.5, color="#555555", ls=":", label="Maximum terminal reward: 1.5")
        axes[0].legend(loc="upper left", fontsize=9)
        for a, row in zip(x, rows):
            axes[0].annotate(f'{row["max_legal_q"]["max"]:.2f}',
                             (a, row["max_legal_q"]["max"]), xytext=(0, 7),
                             textcoords="offset points", ha="center", fontsize=9)
        axes[1].plot(x, [row["hidden_layers"][-1]["ever_active_units"] for row in rows], "o--", color="#236a91")
        axes[1].set(ylabel="Final-layer units active\non at least one probe", ylim=(0, 132))
        axes[2].plot(x, [100 * row["hidden_layers"][-1]["all_zero_fraction"] for row in rows], "o--", color="#8c579c")
        axes[2].set(ylabel="Probes with entire\nfinal layer zero (%)", xlabel="Completed games in mixed-opponent run (0 = inherited parent)", ylim=(0, 55))
        axes[2].set_xticks(x, [f"{a:,}" for a in x])
        for axis in axes:
            axis.grid(alpha=.2)
        fig.suptitle("Saved checkpoints on the same 4,096 observations\nDashed lines join surviving snapshots; changes inside gaps are unknown", fontsize=13)
        path = figure_dir / "checkpoint_timeline.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    if probes:
        online = [probe for probe in probes if probe["online"]]
        x = [probe["local_updates"] for probe in online]
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True, layout="constrained")
        for field, label, color in (("max", "Maximum", "#b44535"), ("p99", "99th percentile", "#bb8d2a"),
                                    ("median", "Median", "#236a91")):
            axes[0].plot(x, [probe["online"]["max_legal_q"][field] for probe in online], label=label, color=color)
        axes[0].axhline(1.5, color="#555555", ls=":", label="Maximum terminal reward")
        target_probes = [probe for probe in probes if probe["target"]]
        axes[0].plot([probe["local_updates"] for probe in target_probes],
                     [probe["target"]["max_legal_q"]["max"] for probe in target_probes],
                     label="Target maximum", color="#626262", ls="--", linewidth=.9)
        axes[0].set(ylabel="Best legal Q\non fixed probes")
        if max(abs(probe["online"]["max_legal_q"]["max"]) for probe in online) > 10:
            axes[0].set_yscale("symlog", linthresh=2)
        axes[0].legend(ncol=3, fontsize=8, loc="upper left")
        axes[1].plot(x, [probe["online"]["hidden_layers"][-1]["ever_active_units"] for probe in online], color="#236a91")
        axes[1].set(ylabel="Final-layer units active\non at least one probe", ylim=(0, 132))
        axes[2].plot(x, [100 * probe["online"]["hidden_layers"][-1]["all_zero_fraction"] for probe in online], color="#8c579c")
        axes[2].set(ylabel="Entire final layer\nzero (%)", ylim=(0, None))
        wx = [window["last_local_update"] for window in windows]
        for axis, metric, label in ((axes[3], "gradient_l2", "Gradient L2 norm"),
                                    (axes[4], "update_l2", "Actual weight-change\nL2 norm")):
            for statistic, color in (("max", "#b44535"), ("median", "#236a91")):
                axis.plot(wx, [window[metric].get(statistic, np.nan) for window in windows],
                          label=f"{statistic.capitalize()} per 100 updates", color=color, linewidth=.8)
            axis.set(ylabel=label, yscale="log")
            axis.legend(fontsize=8, loc="upper right")
        for axis in axes:
            axis.grid(alpha=.2)
        axes[-1].set_xlabel("Optimizer updates since the diagnostic continuation began")
        axes[-1].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        status = "Completed" if final else "Preliminary"
        fig.suptitle(f"{status} diagnostic continuation: {completed_games:,} games\nFixed probes and learning updates ordered in time; no causal conclusion from this plot alone", fontsize=13)
        path = figure_dir / "diagnostic_updates.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def analyze(run_dir, timeline_path, output_dir, figure_dir=None, final=False):
    columns, top, updates_meta = read_updates(run_dir / "updates.jsonl")
    probes_meta, games_meta = {}, {}
    probes = [summarize_probe(record) for record in complete_lines(run_dir / "probes.jsonl", probes_meta)]
    games = [{key: record.get(key) for key in
              ("game", "reward", "capped", "actions", "transitions", "updates", "optimizer_steps", "elapsed_seconds")}
             for record in complete_lines(run_dir / "games.jsonl", games_meta)]
    # Read updates first, then retain only probes/games covered by those updates.
    # Otherwise a fast live run can advance while the large update log is read.
    cutoff_step = int(columns["step"][-1]) if len(columns["step"]) else None
    if cutoff_step is not None:
        probes = [probe for probe in probes if probe["optimizer_steps"] <= cutoff_step]
        games = [game for game in games if game["optimizer_steps"] <= cutoff_step]
    windows = window_summaries(columns)
    intervals = probe_intervals(probes, columns)
    target_intervals = target_changes(probes)
    timeline = summarize_timeline(timeline_path) if timeline_path else None
    completed_games = int(games[-1]["game"]) if games else 0
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "final": final,
        "sources": {"updates": updates_meta, "probes": probes_meta, "games": games_meta},
        "summary": {"completed_games": completed_games, "recorded_updates": len(columns["step"]),
                    "recorded_probes": len(probes),
                    "capped_games": sum(bool(game["capped"]) for game in games),
                    "game_rewards": stats([game["reward"] for game in games]),
                    "metrics": {key: stats(values) for key, values in columns.items() if key not in
                                ("game", "local_update", "step", "elapsed_seconds")}},
        "first_probe": probes[0] if probes else None, "last_probe": probes[-1] if probes else None,
        "top_updates": top,
        "largest_probe_changes": {
            "active_unit_losses": sorted(intervals, key=lambda item: item["active_unit_change"])[:10],
            "zero_fraction_increases": sorted(intervals, key=lambda item: item["zero_fraction_change"], reverse=True)[:10],
            "q_max_increases": sorted(intervals, key=lambda item: item["q_max_change"], reverse=True)[:10],
        },
        "probe_range": {"q_max": stats([probe["online"]["max_legal_q"]["max"] for probe in probes]),
                        "active_last_units": stats([probe["online"]["hidden_layers"][-1]["ever_active_units"] for probe in probes]),
                        "zero_last_fraction": stats([probe["online"]["hidden_layers"][-1]["all_zero_fraction"] for probe in probes])},
        "target_probe_range": {
            field: stats([probe["target"]["max_legal_q"][field] for probe in probes if probe["target"]])
            for field in ("max", "p99", "median")
        },
        "online_minus_target_same_probe": {
            field: stats([probe["online"]["max_legal_q"][field] - probe["target"]["max_legal_q"][field]
                          for probe in probes if probe["target"]])
            for field in ("max", "p99", "median")
        },
        "largest_target_q_increases": sorted(target_intervals, key=lambda item: item["target_q_max_change"], reverse=True)[:10],
        "first_and_last_1000_games": {
            "first": game_window_summary(columns, probes, 1, min(1000, completed_games)),
            "last": game_window_summary(columns, probes, max(1, completed_games - 999), completed_games),
        },
        "activity_excursions": activity_excursions(probes),
        "timeline": timeline,
        "interpretation_limits": [
            "Fixed-probe inactivity is specific to this sample, not proof a neuron never activates on any possible input.",
            "Adjacent probes bound an interval; their difference does not identify the exact update that caused a change.",
            "Large gradients and large optimizer updates are different quantities; recorded associations do not establish causation.",
            "Logs are read at bounded file sizes; preliminary files can end at different completed records while training continues.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "scalar_series.npz", **columns)
    save_csv(output_dir / "scalar_series.csv", columns)
    for name, value in (("windows.json", windows), ("probe_intervals.json", intervals),
                        ("probes_compact.json", probes), ("target_intervals.json", target_intervals)):
        (output_dir / name).write_text(json.dumps(finite_json(value), indent=2, allow_nan=False), encoding="utf-8")
    if figure_dir is not None:
        report["figures"] = make_figures(figure_dir, timeline, probes, windows, completed_games, final)
    (output_dir / "analysis.json").write_text(json.dumps(finite_json(report), indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/diagnostics/20261106/update_run"))
    parser.add_argument("--timeline", type=Path, default=Path("results/diagnostics/20261106/timeline/timeline.json"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/20261106/analysis"))
    parser.add_argument("--figures", type=Path, default=Path("docs/diagnose/figures"))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--final", action="store_true", help="Label a completed, no-longer-growing run as final.")
    args = parser.parse_args()
    report = analyze(args.run, args.timeline, args.output, None if args.no_figures else args.figures, args.final)
    print(json.dumps({"output": str(args.output), "completed_games": report["summary"]["completed_games"],
                      "updates": report["summary"]["recorded_updates"], "probe_range": report["probe_range"]}, indent=2))


if __name__ == "__main__":
    main()
