# DQN updates and game integration

Steps 3 and 4 provide the parts needed to train during one real game:

- A callback runs when each transition is complete. A transition records one
  player's action, its reward, and the observation at their next decision or at
  the end of the game.
- A reusable DQN (deep Q-network) update engine trains a neural network to
  estimate action values.

The code for running many games, evaluating performance, appending to the
saved archive, and saving checkpoints is separate; see [training.md](training.md).
A checkpoint is a file saved so training can resume later. Importing these
modules does not start training.

The ordinary DQN equations follow the [Startups model iteration specification](https://app.notion.com/p/40ce9256a9c08052a173e26d33be00b6).
`training/config.py` also supports Double DQN through `algorithm="double_dqn"`.
See the [training walkthrough](training.md#train-with-double-dqn) for commands.

## Completed-transition callbacks

`collect_episode` accepts an optional async callback,
`on_transition(transition)`. This is a function the collector waits for when a
transition is ready. The **focal player** is the one whose decisions are being
recorded. The callback receives each of their completed `Transition` values
exactly once:

1. At the focal player's first decision, there is no previous transition, so no
   callback runs.
2. At their next decision, the previous action is completed using the current
   observation and legal mask (the list of allowed actions). The callback runs
   and finishes **before** the player selects its next action. A model updated
   there affects that choice.
3. A take-to-play transition uses the actual four-card hand. A play-to-take
   transition waits through the other players' turns; opponent decisions do not
   produce additional focal transitions or model training updates.
4. When the game ends normally, final score settlement supplies the last reward
   and observation. At the action cap, the reward is instead `-1` and the last
   actual observation is used. In either case, the terminal callback runs once,
   with `done=True` and an all-false next mask. It also runs when the final reward
   is zero or an opponent's action ended the episode.

The collector also keeps the transitions locally and returns the whole game's
sequence, called a trajectory, in `EpisodeResult`. It does not change replay
buffers (pools of past transitions used for training) or decide which records
are demonstrations. During an ordinary training game, the callback samples
**existing** replay entries. After collection succeeds, the caller appends the
whole new trajectory to `non_demo`, including wins and capped games.

`TransitionBuilder.complete_pending(observation, mask)` makes this order
possible. It completes and clears the previous action without recording a new
choice. `begin_action` still supports existing callers by completing the
previous action and starting a new one in the same call. The collector calls
it after action selection, when the previous action has already been cleared.
Controllers still receive observation notifications as before. Those
notifications are separate from the training callback and do not create
transitions themselves.

If the game is still unfinished at its action limit, collection ends with an
explicit timeout outcome. `TransitionBuilder.finish_timeout` closes the pending
action with reward `-1`, `done=True` and an all-false next mask. The final callback
runs, then the caller receives the trajectory with `EpisodeResult.capped=True`.
The final state preserves the actual visible board; it can still have a take or
play phase. There is no score settlement or hidden-card reveal, and `game_result`
is `None`. The separate `done` flag makes the DQN target terminal.

If the focal player never acted before a very small cap, there is no pending
transition and no callback for it. The episode still has its timeout outcome.
A game that finishes normally on the last allowed action keeps its normal
reward. Callback and controller errors propagate; they are not converted to
timeout penalties or silently retried.

The callback can be any callable that returns an awaitable (a value that can be
used with `await`), including an async function. A value that cannot be called is
rejected before the game starts. A callable that returns a non-awaitable fails
when it is called. Leave the callback unset when building a buffer or collecting
games without training.

## Optional target clipping

`TrainingConfig(clip_targets=True)` constrains finite training targets to
`[-1.5, 1.5]`. These are the possible total-return bounds: intermediate rewards
are zero and the final reward lies in this range. The default is `False`.

First calculate the ordinary DQN or Double DQN target, including reward and
discount. Then clip the result. Next-action selection still uses raw legal
Q-values. Terminal rewards already satisfy the bounds and are unchanged.

For a raw target of `2.9`, the learner uses `1.5`. An online prediction of `2.9`
therefore has squared error `(2.9 - 1.5) ** 2 = 1.96`. The prediction itself is
**not clipped**, so its gradient still tells the network to reduce it. There
is no additional quadratic regularizer and no change to target-network weight
updates. Clipping targets does not guarantee all predictions stay within bounds
or that predictions inside the range are accurate.

NaN and infinity remain nonfinite and reject an optimizer update. Inspection
with `compute_targets(batch)` returns the targets used for training;
`compute_targets(batch, unclipped=True)` returns the estimates before clipping.
Both are detached and leave learning state unchanged. Optional Q monitoring
saves both summaries when clipping is on, alongside raw network predictions.
See [configuration and commands](training.md#clip-training-targets-to-the-possible-return-range).

## DQN update engine

`training/dqn.py` defines `DQNLearner` and `UpdateResult`. An `UpdateResult`
cannot be changed after creation. `DQNLearner(config)` owns:

- `online`: the `QNetwork` whose weights are changed by training.
- `target`: a separate copy of the initial online network. Its weights are
  frozen for gradient updates, and it is kept in evaluation mode.
- `optimizer`: AdamW, which updates only the online network's parameters. It uses
  the configured learning rate and weight decay (defaults 0.0003 and 0.0001).
- `optimizer_steps`: the number of successful updates, starting at zero and
  continuing across game boundaries.

The learning rate controls the size of optimizer updates; weight decay pulls
weights toward zero during those updates.

Device `auto` uses CUDA (GPU execution) when available and CPU otherwise. An
explicit CUDA request fails if CUDA is unavailable. Initializing a network with
a seed leaves the caller's random-generator state unchanged. If you supply
`online=QNetwork(...)`, the learner owns that network and moves it to the chosen
device. It still makes a separate copy for the target network. Give player
controllers `learner.online.predict` so later training updates affect their
decisions.

`update(batch)` accepts an `ArrayBatch` or the older sequence of `Transition`
values and prepares tensors on the chosen device. Normal training gathers
arrays directly from replay, avoiding a round trip through one Python object
per record. For each record, it trains `Q_online(s, a)`: the online network's
estimated value for the **action stored in that replay record**. The algorithm
setting changes how it calculates the target `y` for a nonterminal record:

```text
algorithm="dqn" (the default):
    y = reward + gamma * max Q_target(next_state, legal_next_action)

algorithm="double_dqn":
    next_action = argmax Q_online(next_state, legal_next_action)
    y = reward + gamma * Q_target(next_state, next_action)

Both algorithms:
    terminal: y = reward
    if clip_targets and y is finite: y = clamp(y, -1.5, 1.5)
    loss = mean((Q_online(state, recorded_action) - y) ** 2)
```

In ordinary DQN, the target network both selects the highest-valued next action
and supplies its value. In Double DQN, the **online network selects** the next
action and the **target network supplies its value**. This separation follows
the [Double DQN paper](https://arxiv.org/html/1509.06461v3).

Both target calculations run under `no_grad`. Gradients come from the
current-state prediction for the recorded action, not from choosing or valuing
the next action. Gamma controls the weight of that future value. With gamma 1
and zero rewards before the game ends, the nonterminal target is the future
value obtained by the selected algorithm, followed by optional target clipping.

Illegal next actions are excluded before selection. Legal negative values stay
negative; masked entries are not treated as zero. Ties choose the lowest action
ID. Neither network evaluates terminal **next states**, so their all-false masks
cannot cause a maximum over an empty set or a calculation such as `0 * -inf`.
Gamma zero also skips next-state inference and uses only the reward.

### Double DQN in a small example

Suppose a replay record leads to a next observation where the predictions are:

| Next action | Legal? | Online prediction | Target prediction |
| --- | --- | ---: | ---: |
| A | Yes | 1.2 | 0.8 |
| B | Yes | 1.0 | 1.3 |
| C | No | 100 | 200 |

With reward 0 and gamma 1:

- Ordinary DQN uses **1.3**, the target network's largest legal prediction.
- Double DQN lets the online network select **A**, then uses the target network's
  value for A: **0.8**.

C is excluded despite its large values. If the current prediction for the
**recorded action at the previous observation** is 0.9, Double DQN's error is
`0.9 - 0.8 = 0.1`. The recorded action and A need not be the same action.

The network architecture, player controller and action map are shared by both
algorithms. Double DQN adds one online-network prediction over the nonterminal
next states in each update; there is still one loss, one backward pass and one
optimizer step. It is intended to reduce overestimation caused by choosing and
valuing an action with the same estimates. It does not constrain Q-values to
the reward range or guarantee that inactive hidden neurons recover.

### Applying the update

Backpropagation calculates how the online weights affect the loss (the mean
squared error above). After this calculation and one AdamW step, the update
counter advances. By default, every 1,000th optimizer update also blends the
**new online weights** into the target weights. This is called a soft update;
tau is the share taken from the online network:

```text
target = (1 - tau) * target + tau * online    # default tau = 0.1
```

The returned `UpdateResult` contains `step`, the single-number `loss` measured
before the update, and `target_updated`. `compute_targets(batch)` returns
targets detached from gradient tracking for tests and inspection. It changes
neither network.

A batch must be nonempty. Normal training uses an `ArrayBatch` from
`ReplayBuffers.sample_array_batch()`, with the configured demo/non-demo mix
(16 + 16 by default). Its arrays own their data and remain stable if the replay
buffers change. DQN checks their shapes, finite values, legal recorded actions
and terminal masks using array operations. It also checks that loss is finite
before changing optimizer state. Invalid caller edits to an array batch are
therefore rejected.

`ReplayBuffers.sample_batch()` and `update(sequence_of_transitions)` remain
available. Starting from the same replay RNG state, both sampling methods select
the same records and leave the same next RNG state. The array path changes data
preparation, not the DQN equations, batch mix, callback timing or number of
updates. Tests and explicit experiments can use other batch sizes; the loss is
always averaged over the batch supplied.

## One-game usage

Run inside the `Startups` conda environment. This example reads the archive,
updates the online model during one game, then adds the complete game's
transitions to the non-demo buffer in **RAM** (working memory). It does not
write the SQLite file or a model checkpoint.

```python
import asyncio
from random import Random

from players.neural import FlatObservationEncoder, NeuralPlayer
from training.config import TrainingConfig
from training.dqn import DQNLearner
from training.episodes import collect_episode, create_episode
from training.replay import load_initial_buffers
from training.runtime import configure_cpu_threads

config = TrainingConfig(seed=42)  # Add algorithm="double_dqn" to use Double DQN.
configure_cpu_threads(config.cpu_threads)
learner = DQNLearner(config)
buffers = load_initial_buffers("buffer_data/ranking_player_15.sqlite3", config)
game_rng = Random(config.seed)
completed_training_games = 0

def focal_factory(player_id, *, seed):
    return NeuralPlayer(
        player_id, learner.online.predict, FlatObservationEncoder(),
        selection_mode="epsilon_greedy",
        epsilon=config.epsilon_for_game(completed_training_games), seed=seed,
    )

async def learn_on_transition(_transition):
    # The completed transition signals one update; it is not inserted yet.
    learner.update(buffers.sample_array_batch())

async def play_one_training_game():
    episode = create_episode(focal_factory, rng=game_rng)
    result = await collect_episode(
        episode, max_actions=config.max_actions_per_game,
        on_transition=learn_on_transition,
    )
    buffers.non_demo.extend(result.transitions)
    return result

result = asyncio.run(play_one_training_game())
completed_training_games += 1
print(result.outcome.reward, learner.optimizer_steps)
```

`create_episode` still chooses each opponent independently, with replacement
(the same strategy may appear more than once). It also randomizes the focal
seat and starting player. The runner handles repeated games, counters,
evaluation and saving to disk. It must not add the same returned trajectory
twice or reset the learner at the start of each game.

## Recovery-file retention

Saving every 200 training games does not require keeping every saved version.
The runner keeps one `latest` recovery checkpoint and replaces it
atomically at the next save: the complete new file replaces the old one in a
single operation. The best model has a separate file.

A 10,000-game run therefore has 50 scheduled recovery writes but keeps only one
latest recovery file. A final save also covers runs that end between scheduled
saves. See [training.md](training.md) for the implemented recovery behavior,
including extra writes around evaluation and replay refreshes. In fixed-pool
mode, `best.pt` keeps the highest valid mean. In growing-pool mode, it keeps the
latest model whose valid mean exceeds the admission threshold.
