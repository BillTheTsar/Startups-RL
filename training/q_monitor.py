"""Optional read-only replay diagnostics at an evaluation boundary.

Sampling uses a local RNG, never the replay, gameplay, evaluation or global RNG.
No optimizer step, model-mode change, gradient clearing or replay insertion is
performed. The runner persists the result with its existing evaluation journal,
so interrupted evaluation recovery reuses the same measurements once.
"""

import hashlib
import json
from math import isfinite
from random import Random

import numpy as np

from .batches import ArrayBatch
from .diagnostics.probes import distribution, measure_network


def monitoring_seed(run_id: str, game: int) -> int:
    """Derive a repeatable diagnostic-only seed without advancing any RNG."""
    return int.from_bytes(hashlib.sha256(f"q_monitor.v1:{run_id}:{game}".encode()).digest()[:8], "little")


def _compact_network(network, batch):
    measured = measure_network(network, batch.states, batch.masks)
    return {
        "legal_q": measured["legal_q"],
        "max_legal_q": measured["max_legal_q"],
        "hidden_layers": [{key: layer[key] for key in (
            "layer", "width", "ever_active_units", "never_active_units",
            "all_zero_rows", "all_zero_fraction", "activation_fraction",
            "positive_units_per_state",
        )} for layer in measured["hidden_layers"]],
    }


def collect_q_monitor(learner, buffers, *, sample_size: int, seed: int) -> dict:
    """Measure both networks on one balanced, uniformly sampled replay batch.

    sample_size is the total requested count. Split it equally between demo and
    non_demo (demo receives an odd extra row); use the other partition to fill
    any shortage. Cap at the number of available rows and sample each partition
    without replacement. The sample changes at later evaluations. Record both
    all-legal-action and best-legal-action Q distributions, hidden activity, and
    actual configured DQN/Double-DQN targets, with terminal targets separated.
    When target clipping is enabled, also report the unclipped targets so the
    constraint cannot hide overestimation. Network measurements are always raw.

    Quantiles exclude nonfinite entries, whose count is reported explicitly.
    They are summaries, never clipping rules or admission criteria. Fractions
    outside the return range use finite values as their denominator. No runtime
    limit is imposed on Q-values and no automatic stopping is introduced.
    """
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("monitor seed must be an integer")
    sizes = {name: len(getattr(buffers, name)) for name in ("demo", "non_demo")}
    total = min(sample_size, sum(sizes.values()))
    if not total:
        raise ValueError("Q monitoring requires occupied replay")
    counts = {"demo": min((total + 1) // 2, sizes["demo"]),
              "non_demo": min(total // 2, sizes["non_demo"])}
    for name in counts:
        counts[name] += min(total - sum(counts.values()), sizes[name] - counts[name])
    rng = Random(seed)
    parts = []
    for name, count in counts.items():
        if count:
            selected = sorted(rng.sample(range(sizes[name]), count))
            parts.append(getattr(buffers, name).arrays_at(selected))
    batch = ArrayBatch._from_replay(**{
        name: np.concatenate([getattr(part, name) for part in parts]) for name in ArrayBatch.FIELD_NAMES
    })
    digest = hashlib.sha256()
    for name in ArrayBatch.FIELD_NAMES:
        digest.update(getattr(batch, name).tobytes())
    targets = learner.compute_targets(batch).detach().float().cpu().numpy()
    report = {
        "version": 1, "requested_sample_size": sample_size, "sample_size": total,
        "partition_counts": counts, "available_rows": sizes, "sampling_seed": seed,
        "sample_sha256": digest.hexdigest(), "architecture_version": learner.config.architecture_version,
        "algorithm": learner.config.algorithm,
        "online": _compact_network(learner.online, batch),
        "target": _compact_network(learner.target, batch),
        "td_targets": {
            "all": distribution(targets, return_bounds=True),
            "terminal": distribution(targets[batch.done], return_bounds=True),
            "nonterminal": distribution(targets[~batch.done], return_bounds=True),
        },
    }
    if learner.config.clip_targets:
        raw = learner.compute_targets(batch, unclipped=True).detach().float().cpu().numpy()
        report.update(
            version=2, clip_targets=True,
            td_targets_unclipped={
                "all": distribution(raw, return_bounds=True),
                "terminal": distribution(raw[batch.done], return_bounds=True),
                "nonterminal": distribution(raw[~batch.done], return_bounds=True),
            },
        )
    # Failing this is a diagnostic bug; no NaN/Infinity is written to metrics.
    json.dumps(report, allow_nan=False)
    return report


def validate_q_monitor(report, config) -> None:
    """Reject malformed/mismatched diagnostics before committing a saved result."""
    fields = {"version", "requested_sample_size", "sample_size", "partition_counts", "available_rows",
              "sampling_seed", "sample_sha256", "architecture_version", "algorithm",
              "online", "target", "td_targets"}
    if config.clip_targets:
        fields |= {"clip_targets", "td_targets_unclipped"}
    expected_version = 2 if config.clip_targets else 1
    if not isinstance(report, dict) or set(report) != fields or type(report["version"]) is not int or report["version"] != expected_version:
        raise ValueError("invalid Q monitoring report")
    if config.clip_targets and report["clip_targets"] is not True:
        raise ValueError("Q monitoring target clipping does not match the configuration")
    for field in ("requested_sample_size", "sample_size", "sampling_seed", "architecture_version"):
        if type(report[field]) is not int:
            raise ValueError("invalid Q monitoring integer")
    if report["requested_sample_size"] != config.q_monitor_sample_size or (
        report["architecture_version"] != config.architecture_version or report["algorithm"] != config.algorithm
        or not 0 < report["sample_size"] <= report["requested_sample_size"]
    ):
        raise ValueError("Q monitoring settings do not match the configuration")
    for field in ("partition_counts", "available_rows"):
        values = report[field]
        if not isinstance(values, dict) or set(values) != {"demo", "non_demo"} or any(
            type(n) is not int or n < 0 for n in values.values()
        ):
            raise ValueError("invalid Q monitoring partition counts")
    if sum(report["partition_counts"].values()) != report["sample_size"] or any(
        report["partition_counts"][p] > report["available_rows"][p] for p in ("demo", "non_demo")
    ):
        raise ValueError("inconsistent Q monitoring sample counts")
    checksum = report["sample_sha256"]
    if type(checksum) is not str or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("invalid Q monitoring sample hash")
    def check_distribution(stats, *, count=None, bounded=True):
        if not isinstance(stats, dict) or any(type(stats.get(key)) is not int for key in ("count", "nonfinite_count")):
            raise ValueError("invalid Q monitoring distribution")
        n, invalid = stats["count"], stats["nonfinite_count"]
        if n < 0 or not 0 <= invalid <= n or count is not None and n != count:
            raise ValueError("inconsistent Q monitoring distribution count")
        expected = {"count", "nonfinite_count"}
        quantiles = ("min", "p01", "median", "p95", "p99", "max")
        fractions = ("below_minus_1_5_fraction", "above_1_5_fraction", "outside_return_range_fraction")
        if n > invalid:
            expected |= {*quantiles, "mean", "std"}
            if bounded:
                expected |= set(fractions)
        if set(stats) != expected or any(type(stats[key]) not in (int, float) or not isfinite(stats[key])
                                        for key in expected - {"count", "nonfinite_count"}):
            raise ValueError("invalid Q monitoring distribution values")
        if n > invalid:
            if [stats[key] for key in quantiles] != sorted(stats[key] for key in quantiles) or stats['std'] < 0:
                raise ValueError("inconsistent Q monitoring percentiles")
            if bounded and any(not 0 <= stats[key] <= 1 for key in fractions):
                raise ValueError("invalid Q monitoring fraction")
        return n

    for network in ("online", "target"):
        if not isinstance(report[network], dict) or set(report[network]) != {"legal_q", "max_legal_q", "hidden_layers"}:
            raise ValueError("invalid Q monitoring network summary")
        check_distribution(report[network]["max_legal_q"], count=report["sample_size"])
        legal_count = check_distribution(report[network]["legal_q"])
        if not report["sample_size"] <= legal_count <= report["sample_size"] * 41:
            raise ValueError("inconsistent Q monitoring legal-action count")
        layers = report[network]["hidden_layers"]
        if not isinstance(layers, list) or len(layers) != 3:
            raise ValueError("invalid Q monitoring hidden layers")
        for index, (layer, width) in enumerate(zip(layers, (256, 256, 128)), start=1):
            integers = ("layer", "width", "ever_active_units", "never_active_units", "all_zero_rows")
            if not isinstance(layer, dict) or any(type(layer.get(key)) is not int for key in integers):
                raise ValueError("invalid Q monitoring activity counts")
            if layer['layer'] != index or layer['width'] != width or (
                not 0 <= layer['ever_active_units'] <= width
                or layer['ever_active_units'] + layer['never_active_units'] != width
                or not 0 <= layer['all_zero_rows'] <= report['sample_size']
            ):
                raise ValueError("inconsistent Q monitoring activity counts")
            for key in ('all_zero_fraction', 'activation_fraction'):
                value = layer.get(key)
                if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("invalid Q monitoring activity fraction")
            if layer['all_zero_fraction'] != layer['all_zero_rows'] / report['sample_size']:
                raise ValueError("inconsistent Q monitoring zero-activity fraction")
            check_distribution(layer.get('positive_units_per_state'), count=report['sample_size'], bounded=False)
    for field in ("td_targets", "td_targets_unclipped") if config.clip_targets else ("td_targets",):
        if not isinstance(report[field], dict) or set(report[field]) != {"all", "terminal", "nonterminal"}:
            raise ValueError("invalid Q monitoring target summary")
        check_distribution(report[field]["all"], count=report["sample_size"])
        terminal = check_distribution(report[field]['terminal'])
        nonterminal = check_distribution(report[field]['nonterminal'])
        if terminal + nonterminal != report['sample_size']:
            raise ValueError("inconsistent Q monitoring target count")
        if field == "td_targets_unclipped" and (
            terminal != report['td_targets']['terminal']['count']
            or nonterminal != report['td_targets']['nonterminal']['count']
        ):
            raise ValueError("inconsistent Q monitoring raw target count")
    try:
        json.dumps(report, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("Q monitoring report must contain finite JSON data") from error
