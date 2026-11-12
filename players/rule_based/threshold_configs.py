"""Five original threshold profiles; chicken mode adds the same rule to each.

Profile       Hand pickup   Combined pickup   Coin pickup   Investment divisor
1                  2              3               3                3
2                  1              3               3                4
3                  1              3               3                3
4                  1              2               2                3
5                  2              4               3                3

Each dictionary overrides ThresholdParameters defaults. A separate chicken flag
avoids maintaining duplicate dictionaries. Creating a player copies and freezes
its parameters. The registry snapshots all profiles on import; restart the CLI
after edits. Display names and profile ordering preserve the original opponents.
"""

from simulator import PlayerId

from .threshold_player import ThresholdPlayer


THRESHOLD_CONFIGS: dict[int, dict[str, int]] = {
    1: {},
    2: {"hand_pickup_threshold": 1, "investment_divisor": 4},
    3: {"hand_pickup_threshold": 1},
    4: {"hand_pickup_threshold": 1, "own_pickup_threshold": 2, "coin_pickup_threshold": 2},
    5: {"own_pickup_threshold": 4},
}


def create_threshold_player(
    player_id: PlayerId, name: str | None = None, *, profile: int = 1, chicken: bool = False,
) -> ThresholdPlayer:
    """Create an ordinary or chicken controller using one of the five profiles.

    Retain the former default names, e.g. 'Threshold player 2' or 'Threshold
    chicken 2'. An explicit name overrides this. All ten combinations share the
    same class and retain no mutable decision state or RNG. Reject unknown IDs.
    """
    if type(profile) is not int or profile not in THRESHOLD_CONFIGS:
        raise ValueError(f"Unknown threshold profile: {profile!r}")
    if name is None:
        name = f"Threshold {'chicken' if chicken else 'player'} {profile}"
    return ThresholdPlayer(player_id, name, parameters=THRESHOLD_CONFIGS[profile], chicken=chicken)
