# WeightedRankingPlayer: scoring and tuning

`WeightedRankingPlayer` scores every action allowed by the action mask (which
marks allowed actions) and chooses the highest score. If several actions tie for
highest, each has an equal chance of being chosen. The player uses its own seeded
random number generator (RNG) for this choice. A seed sets the RNG's starting
point so the same choices can be repeated. Every market slot is a separate
action. Multiple copies of a company card in hand do not create extra invest or
offer actions for that company. A negative score does not make an action illegal.

The code that used to score actions in `RankingPlayer1` is now
`WeightedRankingPlayer`, in `players/rule_based/weighted_ranking_player.py`.
Its original weights remain the
`RankingPlayer1` configuration. Use the new class/import name. To change only
weights, add a configuration rather than a numbered subclass.

## One evaluator, many dictionaries

Named profiles (sets of weights) live in `players/rule_based/ranking_configs.py`:

| Weight | `RankingPlayer1` | `RankingPlayer2` | `RankingPlayer3` |
|---|---:|---:|---:|
| `new_token_value` | -2 | -1 | -3 |
| `dominated_investment_value` | -2 | -1 | -2 |
| `commitment_per_investment` | +1 | +2 | +1 |
| `offer_risk_per_card` | -1 | -1 | -2 |
| `coin_value` | 1/2 | 1/2 | 3/4 |

All other weights are the same across these three profiles. Profile 2 reduces
two investment penalties and gives a larger bonus for collecting companies the
player has already invested in. Profile 3 gives larger penalties for new tokens
and risky offers, and values cash more highly. These settings express preferences;
they do not show that a profile plays better. `RankingPlayer4` keeps its custom
weights in the configuration dictionary. The next fifteen profiles use the same
scoring code with different preferences:

| Profile | Intended emphasis |
|---|---|
| `RankingPlayer5` | Cash opportunist: value coins more and collections less |
| `RankingPlayer6` | Set collector: larger bonuses for matching cards in hand, less weight on deck fees |
| `RankingPlayer7` | Majority pursuer: stronger rewards for visible leads, weaker token penalty |
| `RankingPlayer8` | Minority escape: larger penalties for being behind, smaller pickup bonuses for invested companies |
| `RankingPlayer9` | Token tolerant: no new-token penalty, smaller penalty for investing beyond the investment threshold |
| `RankingPlayer10` | Token shy: stronger token penalty, but protect a valuable clear lead |
| `RankingPlayer11` | Small-company collector: favor holding 5/6 and collecting matching cards |
| `RankingPlayer12` | Large-company collector: favor holding 9/10 and collecting matching cards |
| `RankingPlayer13` | Committed collector: collect companies already invested in, tolerate holdings above the excess threshold |
| `RankingPlayer14` | Flexible collector: no bonus for collecting invested companies, larger penalties above the excess threshold |
| `RankingPlayer15` | Pair builder: reward the first pair more than additional copies |
| `RankingPlayer16` | Triple builder: concentrate the group reward on reaching three copies |
| `RankingPlayer17` | Four-card specialist: concentrate the group reward on reaching four copies |
| `RankingPlayer18` | Protective investor: penalize risky offers strongly, tolerate new tokens |
| `RankingPlayer19` | Surplus trimmer: discourage holdings above the excess threshold and investment beyond the investment threshold |

Each dictionary starts from the original defaults and replaces only the listed
weights. It does not start from the previous numbered profile. Source comments
give the exact weights and their purpose. The labels describe preferences, not
strict rules or measured playing strength. Bonuses for matching cards in hand
also encourage keeping those cards instead of investing them. Profile 17
deliberately rewards a four-card hand even though the required play will break
it up: the player does not score the following move.

The tests compare all actions tied for best on the same positions from seeded
games. This checks that profiles score actions differently, rather than just
choose differently from the same ties. Tests also cover games between copies of
the same profile (self-play) and repeatable games with mixed opponents. They show
different decisions and valid play in the tested cases; they do not show which
profiles win most often. To choose a smaller pool of training opponents, compare
results across opponents, seats, and random seeds. Keep different playing styles
as well as profiles with high individual win rates.

```python
from players import WeightedRankingPlayer, create_ranking_player

# Select a named profile; all profiles construct the SAME Python class.
bold = create_ranking_player(1, "RankingPlayer2", seed=42)
cautious = create_ranking_player(2, "RankingPlayer3", seed=43)
standard = create_ranking_player(3, "RankingPlayer1", seed=44)

# Or supply a one-off dictionary directly. Omitted values keep their defaults.
experiment = WeightedRankingPlayer(
    4, name="My experiment", seed=45,
    weights={"new_token_value": -1, "offer_risk_per_card": -2},
)
```

When a player is created, it copies and checks the dictionary, then stores the
weights in a frozen `RankingWeights` object that cannot be changed. Later edits
to the dictionary do not affect that player. Each player controller owns its
own RNG, even when two use the same configuration. `reset` keeps the weights.

To add a permanent variant, add an entry to `RANKING_CONFIGS` in
`players/rule_based/ranking_configs.py`, for example
`"cash_focused": {"coin_value": 1}`. No new player class or script is needed.
The opponent registry includes every configured name when it is imported.
Restart the command-line interface (CLI) after edits: its registry keeps the
configurations that were loaded when that process started. By contrast,
`create_ranking_player` reads the dictionary each time it is called, so experiments
can also use temporary configurations.

`players/rule_based/registry.py` defines `RULE_BASED_OPPONENTS`, a tuple of named
`OpponentSpec` entries. The CLI makes three independent picks, giving each entry
an equal chance on each pick. It can pick the same entry more than once. Each
ranking configuration is a separate entry, so the pool contains Random, five
Threshold, five ThresholdChicken, and one entry per
configured ranking profile. Pool and lineup output show each profile's name and
weights that differ from the defaults. Each entry creates a fresh player
controller; the registry does not store running players. This replaces the old
class-only `RULE_BASED_PLAYERS` list.

A simple difference in decisions: with a four-card hand `(7, 7, 8, 8)` and no
public investments or known opposing hand cards, profile 1 ties investments and
offers. Profile 2 prefers investing; profile 3 prefers offering. All three use
the same scoring procedure; their weights cause the different choices.

## Selection scores and optional explanations

```text
selection score = position value after + event adjustments
explained gain = selection score - position value before
```

To choose an action, the player does **not** calculate `position_before`. It is
the same for every candidate action, so subtracting it would not change their
order or ties. A take action scores the proposed four-card hand, current
investments, and updated coins. A play action scores the resulting three-card
hand with the played card either invested or offered. Your holdings are your
hand cards plus investments: investing keeps this total unchanged, while
offering reduces it. The player still reads counts from before the action to
check events, such as gaining a new token or already meeting an investment threshold.

`rank_actions` reports these selection scores. Each position component is a
part of the score for the resulting position. For example, with default weights,
`coins` is the resulting balance divided by two, rather than the coins gained.

Call `explain_actions` separately to debug score changes. It scores the current
position as a starting value (the baseline) **once per request**, then subtracts
each baseline component from the matching component of every ranked action.
The results show gains and losses: with default weights, building a pair adds
+1 group value, breaking it loses 1, and removing a penalty can add value.
Action-event adjustments stay unchanged. For a deck draw, subtract the shared
baseline after averaging outcomes, not once per possible card.

The player scores only the immediate action. When taking a card, it does not
consider the play that must follow. Its scoring does not change with the stage
of the game, and it has no special end-of-game scoring rule. The simulator still
reveals hands and settles payments normally at the end.

## Position components

| Breakdown name | Default position value | Relevant counts |
|---|---|---|
| `hand_cards` | Each 5/6: +1; 7/8: 0; 9/10: -1 | Hand only, per card |
| `hand_groups` | 0/1 copies: 0; 2: +1; 3: +3; 4: +5 | Hand only; use one value per company, without adding values for smaller groups |
| `coins` | Coin balance / 2 | Includes negative balances; division keeps fractions |
| `excess_holdings` | -1 per card above `(company_id + 1) // 2` | Hand + investments, per company |
| `majority_position` | Gap <= -2: -4; -1: -2; 0: 0; +1: +1; >= +2: +3 | Gap = (own hand + investments) minus the largest (investments + known hand cards) count among opponents, for that company |

A company with zero own holdings contributes **zero `majority_position` value**.
All opponents count for this comparison, including the token owner. Compare
against the opponent with the largest holding; do not add opponents' holdings
together. Known hand cards give only a minimum count for each opponent. They may
have more, so an apparent lead may not be a lead over their actual cards.

The excess threshold discourages collecting too many cards of one company. For
even company sizes, holding exactly that threshold does not guarantee a lead
over every opponent.

## Action-event adjustments

| Breakdown name | Trigger and default adjustment | Zero cases |
|---|---|---|
| `pickup_commitment` | On a market pickup or possible deck draw, +1 per existing own investment in the acquired company | No investment in that company; all play actions |
| `new_token` | Investing claims an unowned token or strictly overtakes its other owner: -2 | Already own token, merely tie its owner, or do not invest |
| `invested_dominance` | Investing when the **investment count before the action, excluding hand cards**, is already at least `(company_id + 1) // 2`: -2 | Below that threshold before investing, regardless of hand cards; non-invest actions |
| `offer_risk` | After removing the offered card, if remaining own hand + investments do not strictly exceed the strongest eligible opponent's known holding: -1 per remaining own card of that company | Remain strictly ahead, hold zero remaining cards, or do not offer |

For offer risk, compare only with opponents who do **not** own that company's
anti-monopoly token. Count their current investments plus known hand cards;
do not simulate a later pickup. This restriction applies only to `offer_risk`.
The `majority_position` comparison still includes every opponent.

There is **no bonus for matching the token holder's investment count**. There
is no extra bonus for avoiding future fees through token ownership; actual draw
fees already account for exemptions from tokens the player owns. Add event
adjustments to the resulting position value. These penalties can lower an
action's score, but cannot forbid it.

Example of the investment boundary: company 7 has threshold 4. Moving from 3
investments to 4 gets no `invested_dominance` penalty. Moving from 4 to 5 gets
-2. A large hand does not trigger this event by itself.

## Estimating a deck draw

For each company `i`, compute the following. All subtracted card counts refer to
that company:

```text
unaccounted_i = i - own hand cards
                 - all players' public investments
                 - market cards
                 - opponents' known hand cards
P(i) = unaccounted_i / sum(unaccounted copies of all companies)
```

Do not subtract your own known-hand count again: your exact hand already
accounts for those cards. Removed cards and unknown opponent hand cards remain
in this pool. The denominator is **not deck size**. These are estimated draw
probabilities based on what the player can see, not the actual deck contents.

Score each possible fourth card separately, then average each score component
using these probabilities. This matters for bonuses such as hand groups, where
each extra card can change the score by a different amount. Score those bonuses
before averaging. Companies with no unaccounted copies get probability zero.
The calculation does not sample random hypothetical draws.

Drawing costs one coin per individual market entry except entries whose company
token the player owns. This charge applies even if coins become negative. A
market pickup instead collects all coins attached to its selected card. The
commitment bonus applies to both sources.

All scores use Python `Fraction` for exact fractions, including the half-coin
value and the average draw score components. This preserves exact ties. Rounded
values shown for display never determine the action chosen.

## Worked pickup

You have 15 coins, hand `(8, 5, 9)`, and two invested 8s. Another player has three
invested 8s and owns the token; other players have no known 8s. A market 8 has
three coins. Taking it is legal. With default weights, it gives this
**gain/loss explanation**:

| Contribution | Gain |
|---|---:|
| `hand_cards` | 0: an 8 has no inherent hand value |
| `hand_groups` | +1: the hand gains a pair of 8s |
| `coins` | +3/2 |
| `excess_holdings` | 0: four combined 8s do not exceed the threshold of four |
| `majority_position` | +1: combined holdings go from tied at three to ahead by one |
| `pickup_commitment` | +2: two existing investments |
| All other events | 0 |
| **Total** | **11/2** |

Assuming opponents have no known 5s or 9s either, the selection score is **15**:
the resulting position has hand value 0, group value 1, coin value 9, excess
value 0, and majority value 3; the commitment event adds 2. The baseline is
**19/2** (coin value 15/2 plus majority value 2), so the debugging gain is
`15 - 19/2 = 11/2`. Selection never computes that baseline.

For a last-card offer, a company valued at -4 for being behind can improve by
+4 when your holding becomes zero. `offer_risk` is also zero when no own cards
remain. These are separate contributions; neither is a simulated payout.

## Inspecting scores without changing the choice sequence

```python
from action_codec import ActionCodec
from players import WeightedRankingPlayer

player = WeightedRankingPlayer(player_id=1, seed=42)
observation = simulator.get_observation(1)  # During P1's turn.
mask = ActionCodec.make_mask(simulator.get_legal_actions(1))

for entry in player.rank_actions(observation, mask):
    print(entry.action_id, entry.action, entry.score)
    print(dict(entry.contributions))  # Resulting-position values + events.

# Optional debugging only: compute one baseline and show changes relative to it.
for entry in player.explain_actions(observation, mask):
    print(entry.action_id, entry.score)  # Gain, not the absolute selection score.
    print(dict(entry.contributions))    # Position changes + unchanged events.

# Raw current-position values, useful for checking before/after differences:
print(dict(player.position_components(observation)))

# Normal gameplay needs only this call; the inspection above is optional:
action_id = await player.choose_action(observation, mask)
```

Both inspection methods return `ScoredAction` records that cannot be changed.
They list the highest score first and break display ties by action ID from
`ActionCodec`, so the display order is stable. Display order does not affect the
random choice between tied actions. Choosing a single best action does not use
the RNG, and neither inspection method uses it. `reset()` keeps the RNG sequence
and weights; `reset(seed=42)` restarts the sequence. The player stores no past
observations.

## Gameplay and debugging call paths

Normal gameplay uses:

```text
game_loop.run_game -> choose_action -> policy -> _action_value
```

`policy` compares numeric scores in action ID order and keeps only the best IDs.
It does not call `rank_actions`, `explain_actions`, or `position_components`,
and it does not build or sort score reports. `WeightedRankingPlayer` calculates
the numeric score components through `_action_values`. The inspection reports
use the same calculations and add names, so both paths use the same formulas.

`rank_actions` and `explain_actions` run only when a caller explicitly requests
them. Neither the game loop nor the CLI requests them automatically. They can
therefore be left out of training loops. Choosing an action still requires the
score-component calculations and the estimated deck-draw calculation.

When adding a ranking subclass, use `_action_value` to return the score used to
choose an action. The base class's default implementation adds up `_score_action`
contributions. `WeightedRankingPlayer` replaces it with a version that avoids
generating names or reports during play. When changing score components, edit
the shared numeric methods so selection and inspection stay consistent.
Selection scores and explained gains have different meanings when comparing
observations. Do not treat them as interchangeable reinforcement learning (RL)
rewards.

## Tuning and future variants

All numerical settings are in the frozen `RankingWeights` dataclass, a data
object whose fields cannot be changed after creation. Weight tables use these
orders: `hand_values` by company 5..10; `group_values` by count 0..4;
`majority_values` by gap <=-2, -1, 0, +1, >=+2. Each event weight is multiplied
by either its relevant count or a 0/1 value showing whether it happened.
Use negative values for penalties.

```python
from fractions import Fraction
from players import WeightedRankingPlayer

cash_focused = WeightedRankingPlayer(
    1, seed=42, weights={"coin_value": Fraction(3, 4)},
)
```

Integers and Fractions are accepted; floating-point weights are rejected to
preserve exact ties. Weight tables must be tuples of the documented lengths;
unknown keys and invalid values are rejected when the player is created. You
can also pass an existing `RankingWeights` object. Changing a configuration does
not change other players. To change numerical preferences, add a dictionary
entry. To add a new event or score component, change the scoring code.

| Knob | Likely effect of increasing its influence |
|---|---|
| More negative `new_token_value` | More reluctance to claim or take tokens |
| More negative `dominated_investment_value` | More reluctance to invest beyond the existing investment threshold |
| Larger `commitment_per_investment` | More collection of companies already invested in |
| Larger `group_values` | More preference for building and preserving hand combinations |
| More negative `excess_card_value` | Less willingness to collect many cards of one company, even where extra cards could secure a lead |
| More negative `offer_risk_per_card` | More reluctance to offer while remaining tied with or behind an eligible opponent |
| Larger `coin_value` | More preference for attached coins and avoiding deck fees |
| Larger lead rewards / larger minority penalties | More preference for improving the estimated lead or deficit against opponents for a company |

These effects interact. For example, larger group bonuses encourage collecting
but can also discourage investing because investment breaks a hand group.
The majority component changes on pickups/offers, but not when investing keeps
total own holdings unchanged. Scores express rule-of-thumb preferences; they do
not predict final coins. Evaluate weight changes over many seeded games before
drawing conclusions about playing strength.
