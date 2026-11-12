"""Transactional evaluation archives; no player loading, replay, or resume.

One background thread owns its SQLite connection and commits completed records
on a monotonic timer, including while the next game is still running. The
manifest and each committed batch survive independently of derived reports.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import sqlite3
from tempfile import NamedTemporaryFile
from threading import Event, Thread
from time import monotonic
from uuid import uuid4

import evaluation as evaluation_rules
from evaluation import EvaluationAccumulator, EvaluationResult, GamePlan, GameRecord


STORAGE_VERSION = 1
SAVE_INTERVAL_SECONDS = 30.0
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "evaluations"


class ArchiveError(RuntimeError):
    """An archive could not be saved or safely interpreted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def atomic_write_text(path: Path, text: str) -> None:
    """Replace one derived file only after its complete contents have been flushed."""
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_manifest(manifest: dict) -> tuple[str, ...]:
    """Interpret only the supported saved scoring contract, never today's models."""
    if (evaluation_rules.EVALUATION_VERSION != 2
            or tuple(evaluation_rules.PLACEMENT_POINTS) != (1.5, 0.5, -0.5, -1.5)):
        raise ArchiveError("The supported scoring implementation is unavailable; cannot reinterpret saved results")
    version = manifest.get("evaluation_version") if isinstance(manifest, dict) else None
    if type(version) is not int or version not in (1, 2):
        raise ArchiveError("Unsupported evaluation version or saved scoring conventions")
    expected = {
        "evaluation_version": version,
        "players_per_game": 4,
        "placement_points": [1.5, 0.5, -0.5, -1.5],
        "tie_points": "average occupied places",
        "tie_placements": "split credit equally across occupied places",
        "head_to_head": "(wins - losses) / encounters; ties included; unplayed=null; diagonal=0",
    }
    if version == 2:
        if evaluation_rules.TIMEOUT_POINTS != -1.0:
            raise ArchiveError("The timeout scoring implementation differs from the saved contract")
        expected.update(timeout_points=-1.0, timeout_placements="none; rates use all appearances",
                        timeout_head_to_head="excluded; no settled pairwise result")
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
        raise ArchiveError("Unsupported evaluation version or saved scoring conventions")
    for key in ("num_games", "max_actions_per_game"):
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise ArchiveError(f"Invalid manifest {key}")
    try:
        model_ids = tuple(model["model_id"] for model in manifest["models"])
        if (len(model_ids) < 4 or any(not isinstance(item, str) or not item.strip() for item in model_ids)
                or len(set(model_ids)) != len(model_ids)):
            raise ValueError()
    except (KeyError, TypeError, ValueError) as error:
        raise ArchiveError("Invalid model identities in saved manifest") from error
    return model_ids


def _decode_record(payload: str, number: int, model_ids: tuple[str, ...], max_actions: int,
                   evaluation_version: int) -> GameRecord:
    try:
        value = json.loads(payload)
        plan = value["plan"]
        if type(plan["game_number"]) is not int or plan["game_number"] != number:
            raise ValueError("game number differs from its database key")
        if (type(plan["starting_player_id"]) is not int or plan["starting_player_id"] not in (1, 2, 3, 4)
                or len(plan["model_ids"]) != 4 or len(set(plan["model_ids"])) != 4
                or any(item not in model_ids for item in plan["model_ids"])):
            raise ValueError("invalid lineup or starter")
        if (type(plan["game_seed"]) is not int or type(plan["deck_seed"]) is not int
                or len(plan["player_seeds"]) != 4 or any(type(seed) is not int for seed in plan["player_seeds"])):
            raise ValueError("invalid seeds")
        if len(value["scores"]) != 4 or any(type(score) is not int for score in value["scores"]):
            raise ValueError("invalid final scores")
        if type(value["action_count"]) is not int or not 0 < value["action_count"] <= max_actions:
            raise ValueError("invalid action count")
        capped = value.get("capped", False)
        if type(capped) is not bool or (evaluation_version == 1 and capped) or (
            evaluation_version == 2 and "capped" not in value
        ):
            raise ValueError("invalid timeout outcome for the saved scoring version")
        if capped and value["action_count"] != max_actions:
            raise ValueError("timeout must end at the configured action cap")
        return GameRecord(GamePlan(number, plan["game_seed"], tuple(plan["model_ids"]),
                                   plan["starting_player_id"], plan["deck_seed"], tuple(plan["player_seeds"])),
                          tuple(value["scores"]), value["action_count"], capped=capped)
    except (KeyError, TypeError, ValueError) as error:
        raise ArchiveError(f"Invalid saved game {number}: {error}") from error


class RunArchive:
    """A single-producer writer for one new run. Existing archives are never appended.

    append queues immutable, compact outcomes. finish drains them and commits
    the final status in the same transaction. check raises background errors.
    Disk I/O never occurs on the game's event loop after initial creation.
    """

    def __init__(self, manifest: dict, output_root: Path | str = DEFAULT_OUTPUT_ROOT,
                 *, save_interval: float = SAVE_INTERVAL_SECONDS) -> None:
        if isinstance(save_interval, bool) or not math.isfinite(save_interval) or save_interval <= 0:
            raise ValueError("save_interval must be finite and positive")
        self.manifest = json.loads(_json(manifest))
        self._model_ids = _validate_manifest(self.manifest)
        self.save_interval = save_interval
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "Z_" + uuid4().hex[:12]
        self.directory = root / run_id
        self.directory.mkdir()  # Exclusive; never reuse or overwrite an existing run.
        self.database = self.directory / "run.sqlite3"
        self._queue = Queue()
        self._ready = Event()
        self._error: Exception | None = None
        self._closed = False
        self._submitted = 0
        self._thread = Thread(target=self._worker, name="evaluation-archive", daemon=True)
        self._thread.start()
        self._ready.wait()
        self.check()
        try:
            atomic_write_text(self.directory / "manifest.json", json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n")
        except BaseException:
            self.finish("failed", {"message": "Could not export the initial manifest"})
            raise

    def check(self) -> None:
        if self._error is not None:
            raise ArchiveError(f"Saving failed for {self.database}: {self._error}") from self._error

    def append(self, record: GameRecord) -> None:
        self.check()
        if self._closed:
            raise ArchiveError("This archive is closed; runs cannot be resumed or extended")
        number = self._submitted + 1
        payload = _json(asdict(record))
        _decode_record(payload, number, self._model_ids, self.manifest["max_actions_per_game"],
                       self.manifest["evaluation_version"])
        if number > self.manifest["num_games"]:
            raise ArchiveError("Game count exceeds the original run target")
        self._queue.put(("game", number, payload))
        self._submitted = number
        self.check()

    def finish(self, status: str, error: dict | None = None) -> None:
        if status not in ("completed", "interrupted", "failed"):
            raise ValueError("invalid final run status")
        if not self._closed:
            self._closed = True
            self._queue.put(("finish", status, _json(error) if error is not None else None))
            self._thread.join()
        self.check()

    def _commit_batch(self, connection, pending, saved, status="running", error=None):
        count = pending[-1][0] if pending else saved
        if status == "completed" and count != self.manifest["num_games"]:
            raise ArchiveError("Cannot mark an incomplete run completed")
        now = _utc_now()
        with connection:
            connection.executemany("INSERT INTO games(game_number, record_json) VALUES (?, ?)", pending)
            connection.execute(
                "UPDATE run SET saved_games=?, status=?, last_saved_at_utc=?, finished_at_utc=?, error_json=? WHERE id=1",
                (count, status, now, None if status == "running" else now, error),
            )
        return count

    def _worker(self) -> None:
        connection = None
        try:
            connection = sqlite3.connect(self.database, timeout=5)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                PRAGMA user_version=1;
                CREATE TABLE run (
                    id INTEGER PRIMARY KEY CHECK(id=1), run_id TEXT NOT NULL,
                    manifest_json TEXT NOT NULL, status TEXT NOT NULL,
                    saved_games INTEGER NOT NULL, created_at_utc TEXT NOT NULL,
                    last_saved_at_utc TEXT NOT NULL, finished_at_utc TEXT,
                    save_interval_seconds REAL NOT NULL, error_json TEXT
                );
                CREATE TABLE games (
                    game_number INTEGER PRIMARY KEY CHECK(game_number > 0),
                    record_json TEXT NOT NULL
                );
            """)
            now = _utc_now()
            with connection:
                connection.execute("INSERT INTO run VALUES (1, ?, ?, 'running', 0, ?, ?, NULL, ?, NULL)",
                                   (self.directory.name, _json(self.manifest), now, now, self.save_interval))
            self._ready.set()
            pending, saved = [], 0
            deadline = monotonic() + self.save_interval
            while True:
                if monotonic() >= deadline:
                    if pending:
                        saved = self._commit_batch(connection, pending, saved)
                        pending.clear()
                    deadline = monotonic() + self.save_interval
                try:
                    message = self._queue.get(timeout=max(0, deadline - monotonic()))
                except Empty:
                    continue
                if message[0] == "finish":
                    self._commit_batch(connection, pending, saved, message[1], message[2])
                    return
                pending.append((message[1], message[2]))
        except Exception as error:
            self._error = error
        finally:
            if connection is not None:
                connection.close()
            self._ready.set()


@dataclass(frozen=True)
class SavedRun:
    directory: Path
    result: EvaluationResult
    metadata: dict


def load_run(directory: Path | str) -> SavedRun:
    """Recompute supported statistics from a consistent snapshot of saved outcomes.

    Open an existing database (never create one). SQLite may recover a hot
    rollback journal left by a crash; preserve journal files with the database.
    A missing terminal status is reported as unfinished, not inferred completed.
    """
    directory = Path(directory).resolve()
    database = directory / "run.sqlite3"
    connection = None
    try:
        connection = sqlite3.connect(database.as_uri() + "?mode=rw", uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != STORAGE_VERSION:
            raise ArchiveError(f"Unsupported archive storage version: {version}")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ArchiveError("SQLite integrity check failed")
        row = connection.execute("SELECT * FROM run WHERE id=1").fetchone()
        if row is None:
            raise ArchiveError("Archive has no run manifest")
        manifest = json.loads(row["manifest_json"])
        model_ids = _validate_manifest(manifest)
        totals = EvaluationAccumulator(model_ids)
        for number, payload in connection.execute("SELECT game_number, record_json FROM games ORDER BY game_number"):
            if number != totals.completed_games + 1:
                raise ArchiveError("Saved game numbers are not consecutive")
            totals.add(_decode_record(payload, number, model_ids, manifest["max_actions_per_game"],
                                      manifest["evaluation_version"]))
        if totals.completed_games != row["saved_games"] or totals.completed_games > manifest["num_games"]:
            raise ArchiveError("Saved game count does not match the run metadata")
        if row["status"] not in ("running", "completed", "interrupted", "failed"):
            raise ArchiveError("Unknown saved run status")
        if row["status"] == "completed" and totals.completed_games != manifest["num_games"]:
            raise ArchiveError("Completed archive is missing games")
        metadata = {key: row[key] for key in (
            "run_id", "saved_games", "created_at_utc", "last_saved_at_utc", "finished_at_utc", "save_interval_seconds",
        )}
        metadata.update(storage_version=version, status="unfinished" if row["status"] == "running" else row["status"],
                        error=json.loads(row["error_json"]) if row["error_json"] else None)
        return SavedRun(directory, totals.result(manifest), metadata)
    except (sqlite3.Error, OSError, ValueError, KeyError, TypeError) as error:
        raise ArchiveError(f"Cannot read archive {database}: {error}") from error
    finally:
        if connection is not None:
            connection.close()
