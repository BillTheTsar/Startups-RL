"""Read and validate only the archive records selected for a replay buffer.

Selection is uniform over stored rows, before their payloads are inspected.
Invalid selected transitions are skipped without drawing replacements. The
caller owns the read transaction and verifies schemas and saved row bounds.
"""

from random import Random
from typing import TYPE_CHECKING

from .archive import _COLUMNS, _transition

if TYPE_CHECKING:
    from .replay import ReplayBuffer


def _sample_row_ids(connection, partition: str, bounds: dict[str, int],
                    capacity: int, selection_rng: Random) -> list[int]:
    """Reservoir-sample IDs, reading no state or mask payloads.

    Dense positive IDs need no database scan beyond the caller's bounds query.
    When IDs contain gaps, stream only the actual IDs so absent IDs never get
    sampled. Sort the selected IDs to retain chronological FIFO order.
    """
    count, last = bounds["row_count"], bounds["max_row_id"]
    if count == last:
        row_ids = range(1, last + 1)
    else:
        row_ids = (row[0] for row in connection.execute(
            f"SELECT id FROM {partition} WHERE id <= ? ORDER BY id", (last,),
        ))
    selected = []
    total = 0
    for index, row_id in enumerate(row_ids):
        total += 1
        if index < capacity:
            selected.append(row_id)
        else:
            candidate = selection_rng.randrange(index + 1)
            if candidate < capacity:
                selected[candidate] = row_id
    if total != count:
        raise ValueError(f"archive {partition} row count differs from its bounds")
    return sorted(selected)


def sample_archive_partition(
    connection, partition: str, *, bounds: dict[str, int], capacity: int,
    selection_rng: Random, sampling_seed: int,
) -> tuple["ReplayBuffer", int]:
    """Build a buffer from selected records in an already-open read snapshot.

    Check each selected transition's vector, mask, action, reward and terminal
    fields. A record rejected by that validation is omitted without replacement.
    No unselected payload or whole-episode chain is read or validated here.
    Database/query failures and inconsistent row bounds still raise errors.

    Return the new buffer and the archive population count, including any rows
    skipped after selection. The caller decides whether the valid buffer has
    enough entries to train. Its configured capacity is retained in all cases.
    """
    from .replay import ReplayBuffer

    if partition not in ("demo", "non_demo"):
        raise ValueError("partition must be demo or non_demo")
    buffer = ReplayBuffer(capacity, seed=sampling_seed)
    selected = _sample_row_ids(connection, partition, bounds, capacity, selection_rng)
    # Stay below SQLite's commonly supported parameter limit and bound the
    # temporary payload allocation even for a large replay capacity.
    for start in range(0, len(selected), 500):
        chunk = selected[start:start + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"SELECT id, {_COLUMNS} FROM {partition} "
            f"WHERE id IN ({placeholders}) ORDER BY id", chunk,
        ).fetchall()
        if [row[0] for row in rows] != chunk:
            raise ValueError(f"archive {partition} selected rows differ from its bounds")
        for row in rows:
            try:
                transition = _transition(row[1:])
            except ValueError:
                continue
            buffer.append(transition)
    return buffer, bounds["row_count"]
