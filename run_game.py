"""Play Startups in the terminal: python run_game.py [--seed INTEGER].

Each game lists the selected opponents, samples three independently with
replacement, randomizes the human seat and starter, and offers a new game
after scoring. The printed game seed reproduces setup and bot randomness when
the same human decisions are supplied. Each subsequent game gets a fresh seed.
The default pool remains all rule-based players. --catalog-file selects a fixed
pool that may also contain permanently exported neural players.
"""

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from secrets import randbits

from game_loop import GameResult, GameStep, run_game
from players import HumanPlayer, Player
from players.cli_input import console_session, read_line
from players.catalogue import ModelSpec, load_pool
from players.rule_based.registry import OpponentSpec, RULE_BASED_OPPONENTS
from simulator import PlayerId, Simulator


@dataclass(frozen=True)
class GameSetup:
    """A fresh seeded game, independently constructed controllers, and seat labels."""

    seed: int
    human_id: PlayerId
    starting_player_id: PlayerId
    simulator: Simulator
    players: tuple[Player, ...]


def _validate_opponents(opponents) -> tuple[OpponentSpec | ModelSpec, ...]:
    """Validate a selected pool before sampling or entering the human interface."""
    try:
        opponents = tuple(opponents)
    except TypeError as error:
        raise ValueError("opponents must be a nonempty sequence of player specifications") from error
    if not opponents or any(not isinstance(item, (OpponentSpec, ModelSpec)) for item in opponents):
        raise ValueError("opponents must be a nonempty sequence of player specifications")
    identities = [item.model_id if isinstance(item, ModelSpec) else item.name for item in opponents]
    if len(set(identities)) != len(identities):
        raise ValueError("opponent pool model IDs must be unique")
    return opponents


def create_game(
    seed: int | None = None, *, opponents: Sequence[OpponentSpec | ModelSpec] | None = None,
) -> GameSetup:
    """Sample seats and three named opponents uniformly, allowing duplicates.

    Use a local setup RNG and separate seeds for the deck and each bot. Duplicate
    strategy selections produce separate Player objects and separate RNGs. Do not
    use or reseed the process-wide random generator. No input or output occurs.
    Ranking configurations are separate registry entries despite sharing a class.
    A selected pool can contain even one model: three separate controllers are
    still created. Neural specifications share frozen weights, not seat state.
    """
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer or None")
    if opponents is None:
        opponents = RULE_BASED_OPPONENTS
    opponents = _validate_opponents(opponents)
    seed = randbits(64) if seed is None else seed
    rng = Random(seed)
    human_id = rng.choice(Simulator.PLAYER_IDS)
    starter = rng.choice(Simulator.PLAYER_IDS)
    # Materialize all selections before generating the separate stream seeds.
    opponent_specs = tuple(rng.choice(opponents) for _ in range(3))
    simulator = Simulator(seed=rng.getrandbits(64), starting_player_id=starter)
    selected = iter(opponent_specs)
    controllers = []
    for player_id in Simulator.PLAYER_IDS:
        if player_id == human_id:
            controller = HumanPlayer(player_id, "You")
        else:
            specification = next(selected)
            bot_seed = rng.getrandbits(64)
            controller = specification.create(player_id, seed=bot_seed)
        if not isinstance(controller, Player) or controller.player_id != player_id:
            raise ValueError("factories must return a Player for the requested seat")
        controllers.append(controller)
    return GameSetup(seed, human_id, starter, simulator, tuple(controllers))


def format_opponent_pool(opponents: Sequence[OpponentSpec | ModelSpec] | None = None) -> str:
    """List named opponents, their thresholds, and ranking weight overrides."""
    default_pool = opponents is None or opponents is RULE_BASED_OPPONENTS
    if opponents is None:
        opponents = RULE_BASED_OPPONENTS
    opponents = _validate_opponents(opponents)
    lines = ["\nAvailable rule-based opponents" if default_pool else "\nSelected opponents"]
    for number, specification in enumerate(opponents, start=1):
        lines.append(f"  {number}. {specification.name}: {specification.description}")
    lines.append("Three independent uniform draws; duplicate strategies are allowed.")
    return "\n".join(lines)


def _label(player_id: PlayerId, names: Mapping[PlayerId, str]) -> str:
    return f"P{player_id} ({names[player_id]})"


def format_lineup(setup: GameSetup) -> str:
    """Show the sampled seats, randomized starter, and reproducible seed."""
    names = {player.player_id: player.name for player in setup.players}
    lines = [f"\nGame seed: {setup.seed}", "Lineup (turn order: P1 -> P2 -> P3 -> P4 -> P1)"]
    lines.extend(f"  {_label(pid, names)}" for pid in Simulator.PLAYER_IDS)
    lines.append(f"Your seat: P{setup.human_id}. Starting player: {_label(setup.starting_player_id, names)}.")
    return "\n".join(lines)


def format_step(step: GameStep, names: Mapping[PlayerId, str]) -> str:
    """Describe public action effects using only the observer's two observations.

    Deck draws never reveal a company. Market indices refer to the before view.
    Coin payments and token transfers are public. Settlement is reported later,
    so final balance changes are not attributed to the final investment/offer.
    """
    action = step.action
    actor = _label(step.player_id, names)
    before_players = {player.player_id: player for player in step.before.players}
    after_players = {player.player_id: player for player in step.after.players}
    if action.kind == "draw_deck":
        paid = before_players[step.player_id].coins - after_players[step.player_id].coins
        lines = [f"{actor} drew a face-down card; paid {paid} coin(s) to the market."]
        if step.after.deck_size == 0:
            lines.append("  The deck is empty. This player completes the play action, then the game ends.")
    elif action.kind == "take_market":
        card = step.before.market[action.market_index]
        lines = [
            f"{actor} took market card #{action.market_index + 1} (company {card.company_id}); "
            f"collected {card.coins} coin(s)."
        ]
    elif action.kind == "invest":
        lines = [f"{actor} invested one company {action.company_id} card."]
    else:
        lines = [f"{actor} offered one company {action.company_id} card to the market."]
    old_owners = dict(step.before.anti_monopoly_owners)
    for company, owner in step.after.anti_monopoly_owners:
        if owner != old_owners[company]:
            old_label = "unclaimed" if old_owners[company] is None else f"P{old_owners[company]}"
            new_label = "unclaimed" if owner is None else f"P{owner}"
            lines.append(f"  Company {company} Anti-Monopoly chip: {old_label} -> {new_label}.")
    return "\n".join(lines)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a compact aligned text table without terminal-specific control codes."""
    widths = [max(len(row[index]) for row in (headers, *rows)) for index in range(len(headers))]
    def line(row):
        return "  " + " | ".join(value.ljust(width) for value, width in zip(row, widths))
    return "\n".join((line(headers), "  " + "-+-".join("-" * width for width in widths), *(line(row) for row in rows)))


def format_results(result: GameResult, names: Mapping[PlayerId, str]) -> str:
    """Explain settlement from revealed final holdings and reconcile official scores.

    This display calculation never changes the simulator. Minority holders lose
    one coin per share per tied leader; each leader gains three per minority
    share. Empty companies and tied leaders with no minority holders pay nothing.
    Raise RuntimeError if the explanation does not match the simulator's results.
    """
    final = result.final_observation
    if final.phase != "finished":
        raise ValueError("Results require a finished observation")
    ids = Simulator.PLAYER_IDS
    companies = Simulator.COMPANY_IDS
    final_players = {player.player_id: player for player in final.players}
    holdings = {pid: {company: final_players[pid].investments.count(company) for company in companies} for pid in ids}
    lines = ["\nGame over", "Final holdings (including revealed hands)"]
    lines.append(_table(
        ("Player", *(str(company) for company in companies)),
        [(_label(pid, names), *(str(holdings[pid][company]) for company in companies)) for pid in ids],
    ))
    payouts = {pid: 0 for pid in ids}
    rows = []
    for company in companies:
        largest = max(holdings[pid][company] for pid in ids)
        leaders = tuple(pid for pid in ids if largest > 0 and holdings[pid][company] == largest)
        minority_shares = sum(holdings[pid][company] for pid in ids if pid not in leaders)
        changes = {
            pid: minority_shares * 3 if pid in leaders else -holdings[pid][company] * len(leaders)
            for pid in ids
        }
        for pid in ids:
            payouts[pid] += changes[pid]
        rows.append((str(company), ", ".join(f"P{pid}" for pid in leaders) or "none", *(f"{changes[pid]:+d}" for pid in ids)))
    lines.extend([
        "", "Company-by-company payouts (net coin changes)",
        "Each minority share costs 1 coin per leader; each leader receives 3 coins per minority share.",
        "Tied leaders do not pay each other. Market cards and their coins are excluded.",
        _table(("Company", "Majority holder(s)", *(f"P{pid}" for pid in ids)), rows),
    ])
    before_coins = {player.player_id: player.coins for player in result.before_final_action.players}
    scores = dict(result.scores)
    if any(before_coins[pid] + payouts[pid] != scores[pid] for pid in ids):
        raise RuntimeError("Settlement explanation does not match the simulator's final scores")
    lines.extend([
        "", "Final scores",
        _table(("Player", "Before settlement", "Payout", "Final"), [
            (_label(pid, names), str(before_coins[pid]), f"{payouts[pid]:+d}", str(scores[pid])) for pid in ids
        ]),
        "Winner(s): " + ", ".join(_label(pid, names) for pid in result.winner_ids),
    ])
    return "\n".join(lines)


async def pause_after_turn(final: bool = False) -> None:
    """Wait for an empty Enter submission after an opponent's complete turn."""
    prompt = "Press Enter to see the results: " if final else "Press Enter to continue: "
    with console_session():
        while (await read_line(prompt)).strip():
            print("Press Enter without entering a menu number.")


async def ask_play_again() -> bool:
    """Offer a fresh game through a numbered menu; propagate interruption or EOF."""
    with console_session():
        print("\n1. Play another game with freshly sampled opponents\n2. Quit")
        while True:
            choice = (await read_line("Choose [1-2], then press Enter: ")).strip()
            if choice in ("1", "2"):
                return choice == "1"
            print("Please enter 1 or 2.")


async def play_cli(
    seed: int | None = None, *, opponents: Sequence[OpponentSpec | ModelSpec] | None = None,
) -> None:
    """Resample each game from one fixed pool, loaded once for this session."""
    default_pool = opponents is None or opponents is RULE_BASED_OPPONENTS
    if opponents is None:
        opponents = RULE_BASED_OPPONENTS
    opponents = _validate_opponents(opponents)
    while True:
        print(format_opponent_pool() if default_pool else format_opponent_pool(opponents))
        setup = create_game(seed) if default_pool else create_game(seed, opponents=opponents)
        names = {player.player_id: player.name for player in setup.players}
        print(format_lineup(setup))

        async def show_step(step: GameStep) -> None:
            print(format_step(step, names))
            if step.before.phase == "play" and step.player_id != setup.human_id:
                await pause_after_turn(final=step.after.phase == "finished")

        result = await run_game(setup.simulator, setup.players, viewer_id=setup.human_id, on_step=show_step)
        print(format_results(result, names))
        if not await ask_play_again():
            return
        seed = None


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the optional first-game seed and stop cleanly on Ctrl+C or EOF."""
    parser = argparse.ArgumentParser(description="Play Startups against three uniformly sampled opponents.")
    parser.add_argument("--seed", type=int, help="reproduce the first game's setup and bot randomness")
    parser.add_argument("--catalog-file", help="JSON file selecting rule-based and exported neural opponents")
    parser.add_argument("--opponent-device", help="neural opponent device: cpu (default), auto, cuda, or cuda:<index>")
    args = parser.parse_args(argv)
    if args.opponent_device is not None and args.catalog_file is None:
        parser.error("--opponent-device requires --catalog-file")
    try:
        opponents = None if args.catalog_file is None else load_pool(
            args.catalog_file, device=args.opponent_device or "cpu")
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))
    try:
        asyncio.run(play_cli(args.seed) if opponents is None else play_cli(args.seed, opponents=opponents))
    except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
        print("\nGame session stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
