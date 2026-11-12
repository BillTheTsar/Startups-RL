# Building the initial training dataset

Run from the repository root:

```powershell
python -m training.build_buffer
```

This creates `buffer_data/ranking_player_15.sqlite3` with **at least 20,000**
transitions from one designated **RankingPlayer15** per game. A transition is a
training record that connects one of that player's decisions to their next
decision, or to the end of the game. The designated player is called the
**focal player** below.

Every saved trajectory has an ending, either normal settlement or the action
cap, so the final total can exceed 20,000. The command
never overwrites an existing output file. It needs no additional dependencies.

For a small experiment or a dataset with a different filename:

```powershell
python -m training.build_buffer --transitions 100 --seed 72 --output buffer_data/example.sqlite3
```

The focal seat and starting player are chosen separately at random, with each
seat equally likely. Each of the three opponents is also chosen independently,
with every strategy in `RULE_BASED_OPPONENTS` equally likely. This is the same
pool used by the command-line interface (CLI). Sampling is **with replacement**:
the same strategy can be chosen more than once, including RankingPlayer15.
Each seat gets a new controller object. Only the focal player's actions are
recorded.

## Division of responsibilities

| Module | Responsibility |
| --- | --- |
| `training/transitions.py` | `Transition` validates and holds one training record. `TransitionBuilder` connects a player's consecutive decisions and completes their last action when the game ends. Neither runs a game loop nor accesses disk. |
| `training/rewards.py` | `get_episode_outcome(scores, player_id)` supplies ordinary placement rewards. `TIMEOUT_REWARD` supplies `-1` when the action cap ends an episode. |
| `training/episodes.py` | `create_episode(focal_factory, ...)` creates the players and game. `collect_episode(episode, ...)` adds recording only around the focal controller and runs `game_loop.run_game`. The factory can also create a neural player. |
| `training/dataset.py` | `TransitionDataset.add_episode`, `save_dataset`, and `iter_transitions` assign complete games to partitions (separate groups), save them, and validate them when loading. A game's sequence of transitions is called its trajectory. |
| `training/build_buffer.py` | `build_dataset` chooses RankingPlayer15, collects at least the requested number of transitions, and saves them. `main` handles CLI options and progress messages. |

This script builds the **dataset saved on disk**. Training uses separate replay
buffers: pools of past transitions held in memory for repeated sampling. These
buffers and the Q-network, which estimates the value of each action, are
documented in [Neural training components](neural_training_components.md).
The [DQN (deep Q-network) update engine](dqn_updates.md) calculates learning
targets and updates the network. The [multi-game training runner](training.md)
evaluates performance, promotes selected evaluation games to demonstrations,
and supports recovery after interruption.

`build_dataset` and `collect_episode` are asynchronous functions that callers
must await. The CLI calls `asyncio.run` once. The reusable components do not
print messages or prompt for input.

## What one transition means

An episode is one game, ended by normal settlement or the action cap. A normal
terminal observation follows final settlement. A timeout instead retains the
last actual observation, without revealing hands or settling coins. Each `Transition` has these
fields, which cannot be changed after construction:

| Field | Meaning |
| --- | --- |
| `state` (`s`) | Focal player's current observation encoded as 332 float32 (32-bit floating-point) values. |
| `action_id` (`a`) | Selected action, integer 0–40, using `ActionCodec` version 1. |
| `reward` (`r`) | Zero except for the final focal action, which receives placement points or timeout reward `-1`. |
| `next_state` (`s'`) | Same player's next decision observation, their settled terminal observation, or their last visible state at timeout. |
| `done` (`d`) | Whether this transition ends the episode. |
| `mask` (`m`) | 41 true/false values showing which actions are legal; the selected action must be enabled. |
| `next_mask` (`m'`) | Legal actions at the next decision; all false when the game ends. |

States and masks are stored in memory as tuples that cannot be changed. When a
transition is created, states are rounded to float32 and mutable inputs are
copied. Later encoder calls or changes to the caller's inputs cannot change the
saved record. States use the existing [332-value encoder](neural_observation_encoder.md);
the mask stays separate.

The state `s` and action `a` alone do not tell us the next state `s'`. The builder
must wait for what actually happens:

1. At the focal player's take decision, remember `(s, a, m)`.
2. At their play decision, complete the take transition with the **actual
   four-card hand**, reward zero and that play mask; remember the play action.
3. After the other three players' turns, use the focal player's next take
   decision to complete the preceding play transition.
4. If the game ends first, complete whichever focal action is still pending
   with that player's **settled terminal observation**, final reward, `done=True`
   and an all-false next mask. This also applies when their own play ends the game.
   If the action cap ends the episode instead, use the last actual observation
   and reward `-1`, with the same terminal flag and empty next mask.

`TransitionBuilder.begin_action` returns the previous completed transition, or
`None` for the first decision. `complete_pending` can complete the previous
action before the next action is chosen, allowing learning during a game.
`finish_episode` returns the normal final transition; `finish_timeout` closes a
capped episode. `reset` clears any pending work.

Observation callbacks still notify the underlying player after every step,
but do not create extra transitions. An optional async callback,
`collect_episode(..., on_transition=...)`, receives each completed transition
exactly once: before the next focal choice, after final score settlement, or at
the timeout ending.
By default, the collector does no learning and adds nothing to replay buffers.
Hidden cards and the Simulator's complete snapshots never enter the collector.

These records cover **player decision intervals**, rather than single Simulator
steps. A play-to-take interval normally includes opponents' actions. If a
discount factor is later used to weight future rewards, it will apply once per
focal decision interval unless a separate duration feature is deliberately
added. No discount factor is applied while creating this data.

## Rewards and partitions

Final balances determine rank. First through fourth receive **1.5, 0.5, -0.5,
-1.5** points. Tied players receive the average points for the places they occupy.
For example, two tied for first receive 1 each, tied middle players receive 0,
and a four-way tie gives everyone 0. Negative final balances are valid inputs.

When the action cap ends a game, every player instead receives reward **`-1`**
and no one wins outright. These are reward points, not coin balances. Ordinary
placement rewards sum to zero across four players; timeout rewards sum to `-4`.
Replay stores one focal player's experience and does not require a zero-sum
outcome. Only the focal player's trajectory is saved for each collected game.

- **`demo`**: records selected as high-quality demonstrations. Only this
  partition supplies the demonstration buffer `B_demo`.
- **`non_demo`**: all other records, including ordinary model-training games
  **regardless of whether the model wins**. This partition alone supplies the
  separate ordinary buffer `B`. Each training batch will contain 16 samples
  from each pool; demonstrations are not loaded into both pools.

The [training runner](training.md) separately selects outright wins from
qualifying evaluations for `demo`. During these evaluations, the model's
weights are frozen (held fixed). These games are separate from ordinary
model-training games.

The labels say whether records can be used as demonstrations; they do not mean
"win" and "loss". The initial RankingPlayer15 builder puts only its outright
wins in `demo`: its final balance must be strictly greater than every opponent's,
giving a terminal reward of 1.5. Its other games, including tied first and
timeouts, go into
`non_demo`. This rule is in `build_buffer.py`. The dataset and loader do not use
rewards to decide whether records are demonstrations. The existing 20,000
transitions keep their original assignments: 8,350 `demo` and 11,650 `non_demo`.

Future collection code can explicitly choose the partition without changing
the outcome or reward:

```python
dataset.add_episode(model_episode.transitions, partition="non_demo")  # Even after a win.
```

The partition is chosen **after the game finishes**. In both tables, transitions
before the end of the game have zero reward. A final reward of zero still ends
the episode. Q-learning code must check `done` and use only the final reward in
that case, without adding an estimated next-state value (called bootstrapping).
It must never try to find a maximum over an all-false terminal mask. For
nonterminal transitions, it must exclude illegal actions from the next-state
maximum.

## File format and loading

The SQLite file has `metadata`, `demo`, and `non_demo` tables. Each transition
table has its own row `id` followed by the seven fields above. States are
1,328-byte blocks of raw data (blobs), stored as **little-endian float32**: each
32-bit value stores its least significant byte first. Masks are 41 bytes of 0 or
1. `done` is integer 0 or 1. Rewards use SQLite `REAL`, supporting fractional
values throughout **[-1.5, 1.5]**.

Metadata records only compatibility details for the format, encoder, action
codec, byte order and reward scheme. It does not track experiments.

New archives use reward scheme `placement_points_with_timeout_v2`, which adds
the `-1` timeout outcome. The loader also accepts older `placement_points_v1`
archives. Appending new evaluation experience to one of those archives upgrades
its reward-scheme header in the same transaction; historical rows are not
changed. Use the updated code to read the new header. The transition tables,
332-value encoding and action IDs keep their existing formats.

The code that appends training results to the archive can add `evaluation_blocks`
and `evaluation_episodes` tables. These track promotion and recovery so repeating
an operation does not duplicate its saved work (idempotence). They do not change
the transition format or original rows. The dataset loader still reads only the
named transition partitions.

The renamed partitions use file-format version 2. The existing dataset was
updated in place, preserving every transition value and the row order. Encoder
and action-codec versions remain 1. The loader rejects older version-1 files
that have not been migrated because their format is incompatible.

```python
from training.dataset import iter_transitions

path = "buffer_data/ranking_player_15.sqlite3"
for transition in iter_transitions(path, "demo"):
    pass  # B_demo source.
for transition in iter_transitions(path, "non_demo"):
    pass  # B source.
```

The loader opens the file read-only. It checks metadata, masks, state vectors,
rewards, and that each episode is complete with transitions that connect to one
another. It returns transitions in stored order. Each complete episode occupies
consecutive rows in its table, ending at `done=True`. Reading across partitions
does not reproduce the original order of games.

`training.replay` loads the archive without changing it, samples records to fit
buffer capacity, stores them in first-in, first-out (FIFO) order, and samples
small random training batches (minibatches). `iter_transitions(path)` still
visits both tables for inspection.

By default, a game may run for up to 10,000 Simulator actions. A game that ends
normally on the last allowed action keeps its normal reward. Otherwise the
episode ends with `-1`, and its nonempty focal trajectory is saved in `non_demo`.
Use `--max-actions` to change the limit.

With a very small cap, the focal player might never get a decision. Such an
episode has an outcome but no focal transitions, so the builder skips it. After
100 skipped empty episodes, it stops without writing a dataset; this prevents
a build from retrying forever without collecting data. The existing
`--max-discarded-games` option controls this allowance. Controller errors are
raised to the caller rather than silently discarding the game.

## Verification

```powershell
python -B -m unittest discover -s tests -p "test_training*.py"
python -B -m unittest discover -s tests
```

Tests cover the actual take/play/next-turn boundaries, games ending on the
focal player's or an opponent's action, protection of private player views,
callbacks, sampling with replacement, ties, fractional rewards, keeping whole
games when the requested count is exceeded, timeout rewards and empty focal
episodes, format checks,
saving and loading exact values, winning model games in `non_demo`, and a real
CLI collection with a fixed seed.
