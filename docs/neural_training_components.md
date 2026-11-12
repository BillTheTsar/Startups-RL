# First neural player and replay buffers

Training supports both fixed pools and a
[growing pool of neural opponents](training.md#train-with-a-growing-opponent-pool).
This page covers the neural
player and replay buffers, which hold past game records for repeated use in
training. See [DQN updates and game integration](dqn_updates.md) for how the
network learns, and [training, evaluation and recovery](training.md) for running
many games from the command line, adding qualifying evaluation games to the
training buffers and archive, and saving checkpoint files so training can resume.

## Environment

The environment for running on the GPU (graphics processor) is named `Startups`:

```powershell
conda activate Startups
python -B -m unittest discover -s tests
```

Set your editor's Python interpreter to
`C:\Users\HP\miniconda3\envs\Startups\python.exe`. This environment contains
Python 3.13, NumPy and PyTorch 2.14.1 with CUDA 13.0 support. CUDA lets PyTorch run
on an NVIDIA GPU. The required CUDA runtime libraries and cuDNN (GPU libraries
for neural networks) are included.

GPU inference (running model predictions) and autograd (automatic gradient
calculation for training) have been verified on the NVIDIA GeForce RTX 5080
Laptop GPU. The operations implemented here need no separate CUDA toolkit
installation. The same environment also supports CPU execution.

To install the same direct dependencies on another compatible Windows setup:

```powershell
conda env create -f environment.yml
```

The [environment file](../environment.yml) pins PyTorch's package source to the
official CUDA 13.0 wheel index (the source of its prebuilt packages). This choice
matches this machine's installed driver. The encoder, simulator and rule-based
players need only the Python standard library. Replay's compact storage and
older transition-based interface also work without NumPy; the training batch
path uses NumPy arrays. The Q-network (the neural network that estimates action
values) and `training/dqn.py` require PyTorch.

## Network and controller

`players/neural/q_network.py` defines `QNetwork`. Architecture version 1 remains
the default:

```text
332 -> Linear(256) -> ReLU -> Linear(256) -> ReLU
    -> Linear(128) -> ReLU -> Linear(41)
```

Architecture version 2 adds LayerNorm before each hidden ReLU:

```text
332 -> Linear(256) -> LayerNorm(256) -> ReLU
    -> Linear(256) -> LayerNorm(256) -> ReLU
    -> Linear(128) -> LayerNorm(128) -> ReLU
    -> Linear(41)
```

Select it with `QNetwork(architecture_version=2)` in Python, or
`TrainingConfig(architecture_version=2)` for training. LayerNorm rescales the
hidden features separately for each observation, then applies learned scales
and offsets. Version 2 uses `eps=1e-5` and trainable scale and bias. It has no
running batch statistics and behaves the same during training and inference.
The final output layer is not normalized. LayerNorm is a stability measure, not
a guarantee that predicted values remain within the possible reward range.

The numbers are layer sizes. `Linear` is a fully connected layer, and `ReLU`
keeps positive values while replacing negative values with zero. Inputs use the
existing [version-1 encoded observation](neural_observation_encoder.md).
Each output is a Q-value: the estimated value of an action, in `ActionCodec`
order. Q-values can be negative. There is no softmax conversion to probabilities.
The legal-action mask (which actions are allowed) stays separate and is not a
network input.

`forward` accepts one vector or a batch and keeps gradient tracking available
for training. `predict` takes one encoded observation, adapts it to the network's
device and numeric type (`dtype`), and returns 41 Python floats without
gradients. It does not change shared parameters, existing gradients or
training/evaluation mode flags. Neither architecture uses dropout (randomly
disabling units during training) or batch normalization (using statistics
across observations in a batch).

```python
import torch
from players.neural import FlatObservationEncoder, NeuralPlayer
from players.neural.q_network import QNetwork

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = QNetwork(architecture_version=2).to(device)
player = NeuralPlayer(
    1, network.predict, FlatObservationEncoder(),
    selection_mode="epsilon_greedy", epsilon=0.5, seed=42,
)
# The existing game loop can now call await player.choose_action(observation, mask).
# These initial weights are random; no training has occurred.
```

Both architectures use the same inputs, action mapping and player controller.
Checkpoints and permanent neural exports record the selected architecture and
rebuild it on loading. Existing version-1 exports keep their model identities.
Exact resume and continuation preserve the source architecture; start a fresh
run when switching to version 2. See the
[LayerNorm training command](training.md#train-with-layernorm-architecture-version-2).

`NeuralPlayer.choose_action` checks the observation and mask, calls `policy`,
then checks the selected action ID. It does not update the simulator or
automatically call `on_observation`. The policy chooses actions in two modes:

- `greedy`: choose the highest legal Q-value; ties choose the lowest action ID.
- `epsilon_greedy`: with probability epsilon, choose at random with every legal
  ID equally likely; otherwise choose greedily. This random choice is called
  exploration. It can pick the greedy action too, and does not run the model.
  At epsilon zero, selection never calls the random number generator (RNG).

Illegal actions are always excluded. Greedy selection, also called exploitation,
requires 41 model outputs and finite real values for every legal action. Masked
entries are ignored. The old, unimplemented logits/temperature interface for
probability-based selection has been replaced by `QValueModel` and `epsilon`.
A caller can change `player.epsilon`.

Each controller has its own RNG, so controllers make independent random choices
even when they share a network. `reset(seed)` reseeds the RNG. `reset()` keeps
its current state and all references to the model and settings. This first
controller keeps no history, but its observation callback is available for
future models that need past observations. `checkpoint_id` is only a label;
setting it does not load model weights.

## Configuration and replay

`training/config.py` defines `TrainingConfig`. Its values are checked when it is
created and cannot be changed afterward. In the settings below, the optimizer
updates model weights, the learning rate controls update size, and weight decay
pulls weights toward zero. The loss measures prediction error. Gamma controls
the weight of future value estimates, and tau controls how much of the online
network is copied into the target network at each target update. See
[DQN updates](dqn_updates.md) for the exact equations.

Defaults:

| Setting | Value |
| --- | --- |
| Learning algorithm | Ordinary DQN (`algorithm="dqn"`); select `"double_dqn"` explicitly |
| Architecture | Version 1: 332, 256, 256, 128, 41 with hidden ReLU; select version 2 for LayerNorm before each ReLU |
| Q-value monitoring | Disabled (`q_monitor_sample_size=None`); use a positive total sample count to report at evaluations |
| Optimizer / learning rate / weight decay | AdamW / 0.0003 / 0.0001 |
| Loss / discount | Mean squared error / gamma 1 |
| Target update | tau 0.1 every 1,000 optimizer updates |
| Training games | 10,000 |
| Exploration | epsilon 0.5 to 0.01 over 10,000 completed games |
| Demo / non-demo capacities | 20,000 / 40,000 transitions |
| Batch | 16 demo + 16 non-demo |
| PyTorch CPU threads | 1; configurable with `cpu_threads` |
| Evaluation | 200 fresh games every 200 training games |
| Promotion qualification | Evaluation mean strictly greater than 0.45 |
| Opponent pool | Fixed; use `opponent_pool_mode="growing"` for automatic admissions |
| Growing-pool admission | Evaluation mean strictly greater than 0.6, including `-1` rewards for capped games |
| Archive refresh | Replace both RAM buffers every 5,000 completed training games |
| Latest recovery save | Every 200 training games, regardless of performance |
| Game cap | 10,000 actions |
| Timeout reward | `-1` for every player; the episode counts and can supply `non_demo` experience |

`config.epsilon_for_game(completed_games)` decreases epsilon in a straight line
between the configured endpoints and keeps the result within those limits.
Capped training episodes count toward this schedule. Their final reward is `-1`
and their trajectories enter `non_demo`; evaluation includes the same penalty
in its mean. The encoder still has 332 inputs, with no remaining-action counter.
The versioned `to_dict`/`from_dict` format includes encoder and action-codec
metadata so future checkpoint loaders can check compatibility. Configuration
does not create an optimizer or choose a hardware device; `DQNLearner` does both.

Configuration version 6 adds `clip_targets`, off by default. When enabled,
finite training targets are limited to `[-1.5, 1.5]`; raw predictions and MSE
are unchanged. Older configs load with clipping off. The setting can change
in a new continuation, while exact resume retains it. See
[target clipping](training.md#clip-training-targets-to-the-possible-return-range).

Configuration version 5 added `architecture_version`: `1` by default, or `2` for
LayerNorm. It also adds `q_monitor_sample_size`, disabled by default. Complete
version-1 through version-4 records select architecture 1 with monitoring disabled.
The configuration version describes the JSON schema; it is separate from the
network architecture version.

Optional monitoring evaluates the online and target networks on the same replay
sample without changing training. It reports legal Q-value percentiles, actual
training targets and hidden-layer activity. See
[tracking Q-values](training.md#track-q-values-during-training) for the setting,
saved fields and how to read them.

Configuration version 4 added `algorithm`: `"dqn"` by default, or `"double_dqn"`.
It controls the next-state target calculation in `DQNLearner`, while both
algorithms use the same neural player and support both network architectures. See the
[Double DQN example](dqn_updates.md#double-dqn-in-a-small-example) and
[training commands](training.md#train-with-double-dqn).

The version-3 field `cpu_threads` defaults to `1`. This controls
how many CPU threads PyTorch may use for one operation; it does not run games
in parallel. The runner applies it to the process before play, including CPU
neural opponents. The separate inter-operation setting is left unchanged.
The training and continuation CLIs accept `--cpu-threads N`; see
[CPU threads and faster batches](training.md#cpu-threads-and-faster-batches).

The format also retains the version-2 settings: `opponent_pool_mode` (default `"fixed"`),
`opponent_admission_threshold` (default `0.6`) and
`replay_refresh_interval_games` (default `5000`, or `None` to disable).
Growing mode admits qualified frozen models after evaluation. Periodic archive
sampling works in either pool mode. Complete version-1 configurations and
checkpoints still load with fixed opponents and refresh disabled, preserving
their earlier behavior. Exact resume also leaves these features disabled for
runs saved before their scheduling was implemented, even if the saved settings
already contained the new fields. Complete version-1 and version-2 records load
with `cpu_threads=1`. All version-1 through version-3 records load with
`algorithm="dqn"`; saving writes version 6. Loading does not rewrite the original
files. A new continuation can explicitly select Double DQN, growing pools or
refreshes while preserving its inherited learning state.
See the [training guide](training.md#train-with-a-growing-opponent-pool)
for commands, admission rules and the different meanings of `best.pt`.

`training/replay.py` provides `ReplayBuffer`, `ReplayBuffers` and
`load_initial_buffers`. They store transitions: records connecting one player's
action to their next decision or the end of the game. The demo pool holds
selected demonstrations; the non-demo pool holds the other records.

```python
from training.config import TrainingConfig
from training.replay import load_initial_buffers

config = TrainingConfig(seed=42)
buffers = load_initial_buffers("buffer_data/ranking_player_15.sqlite3", config)
batch = buffers.sample_array_batch()  # ArrayBatch: 16 demo, then 16 non_demo.
# Later, after an ordinary model game ends normally or at its action cap:
# buffers.non_demo.extend(episode.transitions)  # Includes wins.
```

At initialization, each archive partition (`demo` or `non_demo`) fills only its
matching pool, regardless of rewards. If a partition has more records than the
pool can hold, reservoir sampling selects random row IDs: every record has the
same chance, and no record is selected twice. The loader then reads and checks
only those selected records. Invalid selections are skipped without replacement,
so the resulting pool can be smaller than its capacity. Valid selections keep
archive age order so the oldest can be removed first later. This is first-in,
first-out (FIFO) eviction.

Loading checks the archive format and row bounds but leaves unselected payloads
alone. It changes nothing in the database. The original dataset contained 8,350
demo and 11,650 non-demo transitions; qualifying evaluations can grow the archive
after that initial build.

At each configured refresh, both live buffers are replaced using the same
uniform-sampling and FIFO-order rules. The refresh uses a consistent archive
boundary and its own saved RNG. RAM-only experience is discarded; the archive
remains unchanged. Uniform sampling restores the archive's mixture, not an equal
share for every earlier model. See [buffer refreshes](training.md#refresh-both-buffers-from-the-archive)
for timing, recovery and how to disable the feature, and
[archive validation](training.md#archive-validation) for the checks and skipped
records. A refresh replaces neither live pool unless both have enough valid
entries for their configured batches.

Both pools use compact CPU storage: float32 (32-bit floating-point) states,
float64 (64-bit floating-point) rewards to preserve every accepted reward value,
and one-byte IDs, flags and mask values. At their combined default capacity, the
numeric data uses about 157.2 MiB. Adding an entry to a full pool removes the
oldest entry.

Each training batch samples entries with equal probability and without
replacement within each pool's portion of the batch. Sampling leaves entries
in the pool and uses a separate private RNG for each pool. Different batches
can reuse entries. Separate records with equal values are kept separately.
If either pool has too few entries, sampling raises an error before advancing
either RNG.

Normal training calls `sample_array_batch()`. It gathers compact replay data
directly into an `ArrayBatch` from `training/batches.py`, avoiding construction
and repeated validation of one Python `Transition` object per sampled row.
`DQNLearner` checks the arrays together, then transfers them to the selected
device. The fields are:

| Field | Shape and type |
| --- | --- |
| `states`, `next_states` | `[batch_size, 332]`, float32 |
| `actions` | `[batch_size]`, int64 action IDs |
| `rewards` | `[batch_size]`, float64 |
| `done` | `[batch_size]`, boolean |
| `masks`, `next_masks` | `[batch_size, 41]`, boolean |

Each batch owns its arrays. Later buffer writes cannot change the batch, and
changing a batch cannot change replay. The arrays can be edited by a caller,
so DQN validates their contents before each update. This uses array operations
instead of repeated Python checks on individual floats.

The older `sample_batch()` still returns a tuple of immutable `Transition`
values and remains accepted by DQN. A single pool provides `sample_arrays(n)`
and the older `sample(n)`. Both paths draw the same entries in the same order
from the same RNG state. Indexing, iteration and `snapshot()` still return
immutable `Transition` values in oldest-to-newest order.

The faster batch path changes neither the sample probabilities, FIFO order,
batch mix, update frequency nor masks. The checkpoint buffer format is also
unchanged. Removing old replay entries and sampling to fit the initial capacity
may leave only part of a game in a pool; each one-step transition remains valid.
The archive on disk still stores complete games.

The runner replaces one `latest` recovery file at each scheduled save,
regardless of performance. It also replaces that file before and after
evaluation processing, and when the run finishes normally. A separate
best-model file follows the configured pool mode: highest valid mean in fixed
mode, or latest valid model above the admission threshold in growing mode.
The default 10,000-game run has 50 scheduled saves, while keeping only one
latest recovery file. See [training.md](training.md) for how the archive and
replay buffers are brought back into agreement after an interruption, and for
the full checkpoint contents.
