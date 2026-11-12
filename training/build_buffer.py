"""One-off RankingPlayer15 dataset creation: python -m training.build_buffer."""

import argparse
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from random import Random

from evaluation import DEFAULT_MAX_ACTIONS
from players.rule_based.ranking_configs import create_ranking_player

from .dataset import TransitionDataset, save_dataset
from .episodes import collect_episode, create_episode


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "buffer_data" / "ranking_player_15.sqlite3"
DEFAULT_TRANSITIONS = 20_000


@dataclass(frozen=True)
class BuildProgress:
    """Stored-game totals, capped attempts, and attempts with no focal action.

    completed_games counts games whose focal trajectory entered the dataset,
    including capped games. discarded_games counts empty focal trajectories;
    capped_games also includes capped attempts in that empty group.
    """

    completed_games: int
    discarded_games: int
    demo_transitions: int
    non_demo_transitions: int
    capped_games: int = 0

    @property
    def total_transitions(self) -> int:
        return self.demo_transitions + self.non_demo_transitions


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


async def build_dataset(
    output: str | Path = DEFAULT_OUTPUT, *, target_transitions: int = DEFAULT_TRANSITIONS,
    seed: int | None = None, max_actions: int = DEFAULT_MAX_ACTIONS,
    max_discarded_games: int = 100,
    on_progress: Callable[[BuildProgress], None] | None = None,
) -> BuildProgress:
    """Collect whole games until at least the target, then save the two partitions.

    Record only the designated RankingPlayer15 seat. Keep the final game in
    full even if the total passes the target. Finish each game before choosing
    demo/non_demo; intermediate rewards are zero and only the final focal action
    receives placement points or the -1 timeout reward. The opponent pool is the same registry as the
    CLI, sampled independently with replacement. Reusable collection lives in
    episodes.py; the choice of focal strategy and stopping target live here.

    This initial dataset treats RankingPlayer15's outright-winning games as
    demo and its other games as non_demo. That eligibility rule belongs here;
    future collectors can put all model-generated games in non_demo, including
    wins, without changing the dataset or reward calculation.

    Capped trajectories enter non_demo and count as completed dataset games.
    An extremely short cap can end the game before the focal player acts. Such
    an attempt contributes no transitions and increments discarded_games. Abort
    if this empty-trajectory budget is exhausted rather than loop indefinitely;
    policy/code errors are never swallowed.
    Refuse an existing output path, and write nothing until collection finishes.
    """
    _positive(target_transitions, "target_transitions")
    _positive(max_actions, "max_actions")
    _positive(max_discarded_games, "max_discarded_games")
    if on_progress is not None and not callable(on_progress):
        raise ValueError("on_progress must be callable or None")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    dataset = TransitionDataset()
    rng = Random(seed)
    completed = discarded = capped = 0

    def focal_factory(player_id, *, seed):
        return create_ranking_player(player_id, "RankingPlayer15", seed=seed)

    while len(dataset) < target_transitions:
        episode = create_episode(focal_factory, rng=rng)
        result = await collect_episode(episode, max_actions=max_actions)
        capped += int(result.capped)
        if not result.transitions:
            discarded += 1
        else:
            partition = "demo" if not result.capped and result.outcome.outright_win else "non_demo"
            dataset.add_episode(result.transitions, partition=partition)
            completed += 1
        progress = BuildProgress(completed, discarded, len(dataset.demo), len(dataset.non_demo), capped)
        if on_progress is not None:
            on_progress(progress)
        if discarded >= max_discarded_games:
            raise RuntimeError(f"Discarded {discarded} games with no focal actions; no dataset was written")

    save_dataset(output, dataset)
    return progress


def main(argv: list[str] | None = None) -> None:
    """Parse CLI options, print occasional progress, and report the saved totals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transitions", type=int, default=DEFAULT_TRANSITIONS,
                        help="minimum transitions; keep the final game in full (default: 20000)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-discarded-games", type=int, default=100,
                        help="abort after this many games with no focal actions (default: 100)")
    args = parser.parse_args(argv)
    last_discarded = 0

    def report(progress: BuildProgress) -> None:
        nonlocal last_discarded
        if progress.completed_games % 25 == 0 or progress.discarded_games != last_discarded:
            print(f"{progress.completed_games} games, {progress.total_transitions} transitions "
                  f"({progress.demo_transitions} demo / {progress.non_demo_transitions} non_demo), "
                  f"{progress.capped_games} capped, {progress.discarded_games} with no focal actions", flush=True)
        last_discarded = progress.discarded_games

    try:
        progress = asyncio.run(build_dataset(
            args.output, target_transitions=args.transitions, seed=args.seed,
            max_actions=args.max_actions, max_discarded_games=args.max_discarded_games,
            on_progress=report,
        ))
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Saved {progress.total_transitions} transitions from {progress.completed_games} games "
          f"to {args.output.resolve()}\n"
          f"demo: {progress.demo_transitions}; non_demo: {progress.non_demo_transitions}; "
          f"capped games: {progress.capped_games}; "
          f"discarded games with no focal actions: {progress.discarded_games}", flush=True)


if __name__ == "__main__":
    main()
