# Parameterized threshold players

All five ordinary threshold opponents and their five chicken counterparts use
**one `ThresholdPlayer` class**. Given the same observation and allowed actions,
it always makes the same choice; it does not choose randomly. The numeric
settings live in `threshold_configs.py`. Chicken mode tries to avoid gaining a
new anti-monopoly token. It is a separate true/false option and uses the same
numeric settings as ordinary mode.

```python
from players import create_threshold_player, ThresholdPlayer

ordinary = create_threshold_player(1, profile=4)
chicken = create_threshold_player(2, profile=4, chicken=True)

# One-off dictionaries work too; unspecified fields keep profile 1 defaults.
experimental = ThresholdPlayer(
    3, name="Experiment", parameters={"own_pickup_threshold": 4}, chicken=True,
)
```

When a player is created, it copies and checks the configuration dictionary,
then stores it in a frozen `ThresholdParameters` object that cannot be changed.
Later edits to the dictionary do not affect that player. Player controllers do
not have random number generators (RNGs) or store past observations. Reset keeps
the player's identity, settings, and chicken mode. After sampling opponents,
the registry creates fresh controllers; it does not store ten running players.

## Original profiles

| Profile | Hand pickup minimum | Hand + investment pickup minimum | Coin pickup minimum | Investment divisor |
|---|---:|---:|---:|---:|
| 1 | 2 | 3 | 3 | 3 |
| 2 | 1 | 3 | 3 | 4 |
| 3 | 1 | 3 | 3 | 3 |
| 4 | 1 | 2 | 2 | 3 |
| 5 | 2 | 4 | 3 | 3 |

The factory function still names players `Threshold player N` or
`Threshold chicken N` by default. In the opponent registry and command-line
interface (CLI), the names remain `ThresholdPlayer1` through `ThresholdPlayer5`,
followed by `ThresholdChicken1` through `ThresholdChicken5`. Their order and
descriptions are unchanged. Within a fixed opponent pool, each has the same
chance of being sampled as before.

The old numbered classes and modules were removed. Replace calls such as
`ThresholdPlayer4(1)` with `create_threshold_player(1, profile=4)`, and
`ThresholdChicken4(1)` with `create_threshold_player(1, profile=4, chicken=True)`.
Use `.parameters` to inspect settings and `.chicken` to check whether chicken
mode is on. Ordinary and chicken profiles now have the same Python type.

## Unchanged taking rules

1. Look at legal market slots. A slot qualifies if its company meets either the
   minimum count in hand or the minimum combined count in hand and investments.
   Count your cards **before** pickup.
2. If no slot qualifies, look for legal slots with at least the minimum number
   of attached coins.
3. In either group, choose the lowest company ID. Break ties by most coins,
   then lowest market index. Ordinary and chicken modes use the same priorities.
4. Otherwise draw from the deck, regardless of balance or market fees. If an
   action mask (which marks allowed actions) has been restricted by the caller
   and also forbids drawing, reject the request rather than take a card that
   does not qualify.

## Unchanged playing rules

Investing requires `hand_count[company] >= company // investment_divisor`, where
`//` means integer division rounded down. Count only the current four-card hand,
excluding investments. Choose the lowest company ID that qualifies. When
offering, choose the company with the fewest cards in your hand and investments
combined; break ties by highest company ID. The mask excludes offers of the
company just taken from the market. Use the same rules on the final turn.

With `chicken=True`, first exclude investments that would gain a **new**
anti-monopoly token, then apply the hand threshold. Keeping a token you already
own or tying another token owner's public investment count is safe under this
rule. If no safe investment qualifies, offer when legal. If no offer is legal,
allow legal investments that meet the ordinary hand threshold. This can force the
player to gain a new token when a market pickup leaves it holding four cards of
the company just taken. The fallback still requires the hand threshold, even
when the caller restricts the mask.

With a normal four-card hand, ordinary profile 2 always has an investment that
meets its threshold. Chicken profile 2 can still offer, because avoiding new
tokens can rule out those investments.

## Preservation checks

`tests/fixtures/threshold_behavior_before_refactor.json` was captured from the
original ten classes before editing them. Tests check that the new code matches
that saved behavior:

- 59,920 decision/error cases, covering every three- and four-card hand
  combination, including repeated cards but ignoring card order, across several
  market, investment, and token-ownership layouts and restricted action masks.
- Exact action sequences, scores, and winners from 40 seeded games between
  copies of the same profile (self-play).
- Pool display, seeded lineups, and 12 mixed CLI games with the reference
  ranking profiles held fixed. Holding them fixed lets these tests check the
  threshold code change without mixing in separate changes to the ranking pool.

Existing scenario tests still check thresholds, tie-breaking, what players are
allowed to know, token transfers, and the forced-investment exception. These
checks show that behavior matches in the tested cases. They do not cover every
possible game state.
