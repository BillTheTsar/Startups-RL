# Neural observation encoder, version 1

`players/neural/encoder.py` converts one player's `Observation` into a flat
(one-dimensional) sequence of **332 numerical values**. This is the agreed
input format for neural models. The same function accepts observations from
earlier in the game, from another player's turn, and from the end of the game.
It does not depend on the model,
how actions are chosen, the learning objective, or the training framework.

```python
from players.neural import FlatObservationEncoder
from simulator import Simulator

game = Simulator(seed=42)
encoder = FlatObservationEncoder()
vector = encoder(game.get_observation(1))

assert len(vector) == 332
assert vector.typecode == "f" and vector.itemsize == 4
taken_market_company = vector[encoder.SLICES["taken_market_company"]]
metadata = encoder.metadata()
```

The result is a new standard-library `array('f')` containing four-byte
floating-point numbers (float32 on the supported runtime). NumPy and PyTorch
are not required. A model adapter such as `QNetwork.predict` can convert the
array to a tensor, the array format used by the model's framework. Every call
returns its own array. Changing it cannot change another encoded array or the
original observation. Calling `FlatObservationEncoder.encode(observation)`
has the same effect as calling `encoder(observation)`.

The controller formerly in `players/neural.py` now lives in
`players/neural/base.py`. Existing imports such as `from players import NeuralPlayer`
and `from players.neural import NeuralPlayer` still work. Both packages also
export the encoder. See [neural training components](neural_training_components.md)
for the implemented Q-network, action selection that respects the legal-action
mask, and replay buffers that store training examples.

## Completeness audit and deliberate omissions

The vector includes every current board quantity available in `Observation`:

| Observation field | Encoded representation |
|---|---|
| `player_id` | Sets the viewpoint: this player is always self |
| `current_player_id` | Active player relative to self, or none |
| `phase` | The game's take/play/finished phase |
| `hand` | Exact six company counts |
| `players` | All four players' coins, hand sizes, investment counts, and public known-hand counts |
| `market` | Every card's company and attached coins at its original market index |
| `deck_size` | Remaining deck count |
| `anti_monopoly_owners` | Each company's owner, including unowned |
| `taken_company_this_turn` | Encoded under the clearer name **`taken_market_company`** |

The simulator field is still named `Observation.taken_company_this_turn`.
The encoder calls it `taken_market_company`. It records the company of a card
taken from the **market**, which cannot be offered during that turn's play
decision. The value is none before taking, after drawing from the deck, and
after the turn ends. It does not reveal which company was privately drawn.
A turn contains one take and one play.

The phase describes the whole game. If another player is taking a card, the
phase is still `take`; the active-player category identifies that other player.
The phase is `finished` only when the game is complete.

The following quantities can be calculated from the encoded values, so they
do not need separate fields:

- Combined holdings, majority gaps (differences in investment counts), and
  whether an investment would acquire a token can be calculated from counts and
  current token owners. Token ownership is encoded explicitly because a tie
  keeps the previous owner.
- Draw fees can be calculated from individual market entries and token exemptions.
- Unaccounted-for copies are company supply minus the viewer's exact hand, all
  public investments, market cards, and opponents' known hand counts.
  The viewer's public known counts must not be subtracted a second time.
- In an observation produced by the simulator, `phase == play` and
  `deck_size == 0` identify the final play still to be made. The game is not yet
  `finished`.
- Legal choices can be determined from the visible fields. The simulator still
  supplies the definitive 41-entry legal-action mask separately. The mask is
  not a model input.

**Scope:** the vector contains the agreed representation of current quantities.
A strategy that uses game history could still need more information. By explicit
user choice, the encoder discards the order of cards within hands and investment
lists. That order can reveal some of the sequence of events, even when card
counts match. The vector does not include full action timing, earlier
observations, or inferred opponent strategies. Future sequences of observations
can provide more history.

The encoder replaces absolute seat numbers with positions relative to the
viewer, keeping cyclic turn order and all ownership relationships. Player names
and strategy labels are not in the observation, so the encoder does not add them.

The cards in the deck and their order, removed cards, unknown cards held by
opponents, and random number generator (RNG) state are private simulator
information. They are never inputs. The format assumes fixed rules such as
four players, five removed cards, and company supplies 5..10. These rules are
not repeated as values in each observation. Changing the rules may require a
new encoding version.

## Ordering and scaling

- Companies always occur in order **5, 6, 7, 8, 9, 10**.
- Relative players occur in order **self, next, following, last**. For viewer 3,
  the corresponding absolute IDs are **3, 4, 1, 2**. Relative index is
  `(absolute_id - viewer_id) % 4`, regardless of whose turn it currently is.
- A count for company `i` is divided by **`i`**, its total supply. This applies
  to exact hand counts, investments, and public known-hand counts.
- Player balances and market coins are divided by **15**. Balances can be
  negative, and either coin feature can exceed 1. Values are never clipped
  (forced into a fixed range).
- Hand sizes are divided by **4**, and deck size by **28**.
- One-hot entries represent a category with one 1 and all remaining entries 0.
  Presence entries use 1 for an occupied market slot and 0 for an unused slot.
  Both kinds of entry are unscaled zeros or ones.

These divisors are fixed. They are not calculated from a batch of observations
or changed for each game. Float32 causes ordinary rounding (for example, `1/7`
is approximate), but distinct integers in the game's normal ranges remain
distinguishable after division. The encoder keeps no history or changing
statistics for scaling values.

## Exact vector layout

All slices include the start index and exclude the stop index, as in Python.
For example, `0:3` means indices 0, 1, and 2. `FlatObservationEncoder.SLICES` is
a read-only mapping from schema labels to slices.

| Slice / index | Schema label | Meaning |
|---|---|---|
| `0:3` | `phase` | One-hot `[take, play, finished]` |
| `3` | `deck_size` | Deck size / 28; its named slice is `3:4` |
| `4:11` | `taken_market_company` | One-hot `[none, 5, 6, 7, 8, 9, 10]` |
| `11:16` | `active_player` | One-hot `[none, self, next, following, last]` |
| `16:22` | `hand_counts` | Exact viewer hand counts, each divided by company ID |
| `22:78` | `players` | Four consecutive 14-value public-player blocks |
| `78:108` | `anti_monopoly_owners` | Six consecutive 5-value token-owner blocks |
| `108:332` | `market` | Twenty-eight consecutive 8-value market-slot blocks |

Total: **3 + 1 + 7 + 5 + 6 + 4*14 + 6*5 + 28*8 = 332**.

### Public-player blocks

For relative player `r = 0..3`, let `b = 22 + 14*r`:

| Entries | Meaning |
|---|---|
| `b:b+6` | Six investment counts, company order, each divided by company ID |
| `b+6:b+12` | Six public known-hand counts, company order, same scaling |
| `b+12` | Coin balance / 15 |
| `b+13` | Hand size / 4 |

Known counts are **lower bounds**: a player may hold more copies than the public
count shows. Zero means no copies are publicly guaranteed, not that the player
has none. The vector includes the viewer's own public counts because they show
what others know about the viewer. These counts are never replaced with the
exact hand, which has its own `16:22` block.

### Token-owner blocks

For company `i`, let `b = 78 + 5*(i-5)`. Its five values are one-hot:

```text
[unowned, owned_by_self, owned_by_next, owned_by_following, owned_by_last]
```

For viewer 3, a token held by player 1 therefore encodes `[0, 0, 0, 1, 0]`.

### Market-slot blocks

For market index `j = 0..27`, let `b = 108 + 8*j`. Its eight values are:

```text
[present, is_5, is_6, is_7, is_8, is_9, is_10, coins / 15]
```

A company-8 card with three coins is `[1, 0, 0, 0, 1, 0, 0, 0.2]`.
Unused slots contain only zeros, including the presence entry. An occupied slot
is still marked present when it has no coins. Presence does **not** mean the
card can legally be taken. A card stays encoded when the viewer owns its
company's token, even though the viewer cannot take it.

Never sort or merge market entries. Slot `j` matches ActionCodec action ID
`j + 1` in that same observation. Removing a card shifts later market slots.
The next observation and its action mask must use the new indices.

## Historical and terminal views

Use the same viewer for every observation in a sequence:

```python
history = game.get_history_observation(None, player_id=1)
vectors = [encoder(observation) for observation in history]
```

Each snapshot uses only the information available at that time; later knowledge
is not added to earlier snapshots. The encoder does not store observations or
reconstruct/replay game actions. An observation sequence includes individual
steps, with separate take and play decisions, rather than only the viewer's
turns. For a normal policy decision, the active player is always self. The other
active-player categories allow historical views to show other players' steps.

After final scoring, the phase is finished and the active player is none. Exact
hand counts, public known-hand counts, and hand sizes are all zero. Public
investments include the revealed hands. Coin values are the final settled
balances. Market cards and token owners are encoded exactly as supplied by
that final observation.

A training episode can also end at its action cap. In that case, the final
transition uses the last actual observation without settling coins or revealing
hands. Its encoded phase can still be take or play. The separate `done=True`
flag and all-false next mask tell the learner that the episode ended; the input
vector itself does not need an artificial finished state. The encoder remains
332 values and does not include an action counter or remaining budget.

## Validation and checkpoint compatibility

The encoder rejects inputs that are not an `Observation`, invalid IDs/categories,
duplicate or missing player/company records, malformed card records, impossible
per-company counts, a market with more than 28 entries, negative market coins,
contradictory information about the viewer's hand, and coin values the format
cannot represent. It does not silently discard entries or clip values.

The encoder does not check that all cards are accounted for, that a state could
actually be reached through play, or that all phases and rules are consistent.
Those checks remain simulator concerns.

Every call produces a new array, without random sampling or access to the
simulator. Always pass the intended player's `Observation`. The encoder cannot
detect that you accidentally supplied a valid observation from another viewer.

The encoder's checkpoint metadata is:

```json
{
  "name": "startups.flat_observation",
  "version": 1,
  "size": 332,
  "dtype": "float32"
}
```

Save this under a checkpoint's `observation_encoding` key, alongside the
action-codec version and model configuration. Compare the whole record when
loading. Matching the vector length alone is not enough. Reordering categories,
changing scaling, or changing what a feature means requires a new version, even
if the length stays 332. The encoder itself does not load checkpoints or train
models.
