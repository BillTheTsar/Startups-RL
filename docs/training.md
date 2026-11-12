# Training, evaluation and recovery

[Documentation index](README.md)

Training supports fixed and growing opponent pools. Importing the Python modules
does not start it. For automatic admission of new neural opponents, follow the
[growing-pool walkthrough](#train-with-a-growing-opponent-pool).
To select Double DQN, follow the [Double DQN walkthrough](#train-with-double-dqn).
To use LayerNorm, follow [architecture version 2](#train-with-layernorm-architecture-version-2).
To start training, run these commands from the project folder (`D:\Startups`):

```powershell
conda activate Startups
python -m training.train --seed 42
```

You can also start it with one command, without activating the environment first:

```powershell
conda run --no-capture-output -n Startups python -m training.train --seed 42
```

This starts training, evaluates the model at set intervals, and saves checkpoints
(files from which training can resume). After setup, the command-line interface
(CLI) prints the output folder and whether training will use the CPU or GPU.

The defaults are 10,000 completed training games and the existing
`buffer_data/ranking_player_15.sqlite3` archive (the database of saved training
examples). Training uses CUDA, which runs the model on a compatible NVIDIA GPU,
when available. Each run gets a new folder under `results/training/`.
Without a pool file, the starting pool contains the existing rule-based players.
To name that folder yourself or require CUDA:

```powershell
python -m training.train --seed 42 --device cuda --output results/training/first_dqn
```

The output folder must not already exist. `--archive path.sqlite3` selects
another compatible database. Its `demo` and `non_demo` groups must each contain
enough examples for one training batch. During setup, training loads a random
sample from each group, up to that buffer's capacity. This does not change the
database. New configurations also refresh both RAM buffers every 5,000 completed
games by default; see [buffer refreshes](#refresh-both-buffers-from-the-archive).

`--games 1000` changes a fresh run's total. For other settings, save a complete
`TrainingConfig.to_dict()` JSON object and supply `--config config.json`.
`--games`, `--seed`, `--device` and `--cpu-threads` take priority over the matching
JSON settings. CPU operations use one PyTorch thread by default; the faster
array-based replay batches are used automatically.
Epsilon is the chance of choosing a random legal action during training.
Changing the game total does not change how quickly epsilon decreases; its
default decrease still runs over 10,000 completed games.

## Choose the opponents for training

The training configuration and the opponent pool answer different questions:

- `--config` supplies learning settings, such as the learning rate, epsilon
  schedule, replay sizes and number of games.
- `--catalog-file` supplies the starting list of models that can appear as opponents.

You can use either file without the other. Omitting `--config` keeps the default
learning settings. Omitting `--catalog-file` in a fresh run keeps the existing
rule-based pool. A pool file is not a `TrainingConfig` export; it is a separate,
short JSON object containing model IDs.

To include a trained model, first save a permanent neural player from its
checkpoint. Use the source run's folder in place of `YOUR_RUN`:

```powershell
conda activate Startups
python -m players.neural.export_model --checkpoint results/training/YOUR_RUN/best.pt --name "My neural opponent"
python -m players.catalogue --list-models
```

The listing shows everything available: built-in rule-based players and exported
neural players. It does not mean that every listed model will be selected for
training. Copy the full `neural_...` ID of the model you want. The readable name
is for display; the permanent ID identifies its exact weights and playing rules.

Create `configs/training_pool.json` with the following content, replacing the
neural placeholder with that full ID:

```json
{
  "model_ids": [
    "RandomPlayer",
    "ThresholdPlayer1",
    "RankingPlayer7",
    "neural_<full hash>"
  ]
}
```

Each ID appears only once. The pool can contain one or more entries because
training samples opponents with replacement. Its order is recorded and matters
for reproducing seeded choices. Unknown IDs, empty pools and duplicates are
rejected before training starts.

Neural IDs normally refer to `players/neural/weights/` in the project. If the
exports are elsewhere, add `"weights_root": "../saved_neural_players"`, for
example. Relative paths inside this field start from the folder containing the
pool JSON, so that example points from `configs/` to the project's
`saved_neural_players/`. An absolute path also works.

Inspect your selected pool and start training:

```powershell
python -m players.catalogue --list-models --catalog-file configs/training_pool.json
python -m training.train --catalog-file configs/training_pool.json --seed 42 --output results/training/mixed_opponents
```

To also change learning settings, supply both files:

```powershell
python -m training.train --config training_config.json --catalog-file configs/training_pool.json --output results/training/mixed_experiment
```

Each training game independently samples three entries from the selected pool.
All entries have the same chance on each draw, and one model can occupy multiple
opponent seats. The model being trained is the fourth player. Its periodic
evaluations use the same selected opponent pool and the same sampling rule.
There is no separate evaluation-pool setting inside a training run. Use the
[standalone evaluator](evaluation.md#evaluate-a-selected-mixed-pool) with another
pool file when you want a different comparison.

Selected neural opponents are frozen: their weights do not learn alongside the
focal model. Each network is loaded once for the pool, while every seat receives
a fresh controller. Opponents use CPU by default, independently of the learner's
`--device`. To run both the learner and neural opponents on GPU:

```powershell
python -m training.train --catalog-file configs/training_pool.json --device cuda --opponent-device cuda
```

`--opponent-device auto` chooses CUDA when available and otherwise CPU.
Rule-based opponents have no neural device setting.

In the default `"fixed"` mode, the chosen pool stays fixed for the run. In
`"growing"` mode, qualifying versions of the learner can join it after evaluation.
Exporting another model manually or editing the selection file does not change
either kind of running pool.
Recovery stores the actual ordered pool, its rule settings and its neural model
identities, paths and weight fingerprints. The readable metrics and best-model
metadata also identify the opponents used. Keep the selected exported-player
folders at their recorded locations for recovery; the original pool JSON is not
needed to resume.

## Train with a growing opponent pool

Use `opponent_pool_mode="growing"` when you want successful versions of the
learner to become opponents during the same run. The same `training.train` and
`training.continue_training` commands support both modes.

### Start a fresh growing-pool run

With the `Startups` environment active, create a complete configuration:

```powershell
@'
import json
from pathlib import Path
from training.config import TrainingConfig

config = TrainingConfig(
    opponent_pool_mode="growing",
    opponent_admission_threshold=0.6,
    replay_refresh_interval_games=5000,
)
with Path("growing_config.json").open("x", encoding="utf-8") as stream:
    json.dump(config.to_dict(), stream, indent=2)
'@ | python -
```

Choose the starting opponents using the [pool-file walkthrough above](#choose-the-opponents-for-training).
Then run:

```powershell
python -m training.train --config growing_config.json --catalog-file configs/training_pool.json --seed 42 --output results/training/growing_run
```

The configuration-writing snippet refuses to overwrite an existing JSON file.
The training output folder must not already exist. You can omit
`--catalog-file` to start with all built-in rule-based players. Exported neural
players join the starting pool only if selected in the pool file; the catalogue
is the list of available models, not a command to select all of them.

By default, newly admitted models are saved under `players/neural/weights/`.
To choose another destination, add `--weights-root saved_neural_players`.
That option controls **new admissions**. The `weights_root` inside a pool JSON
instead tells the loader where to find that file's **starting neural opponents**.
These locations can differ. Paths supplied on the command line are relative to
the folder where you run the command.

### Continue a trained model with a growing pool

To keep learned weights, optimizer state and replay, use a full recovery
checkpoint and create a new stage. Follow the
[continuation configuration example](#continue-finished-training-with-new-settings)
to copy the parent's settings into a new JSON file. In its `replace(...)` call,
also set:

```python
opponent_pool_mode="growing",
opponent_admission_threshold=0.6,
replay_refresh_interval_games=5000,
```

Save it as `growing_continuation_config.json`, then run:

```powershell
python -m training.continue_training --from-checkpoint results/training/PARENT/latest.pt --config growing_continuation_config.json --output results/training/growing_stage
```

Replace `PARENT` with the source folder. The new stage inherits the entire pool
saved in that checkpoint, including earlier admissions. Add `--catalog-file`
if you want to replace that starting pool instead. The new stage starts its own
game count, evaluation schedule, refresh schedule and best-model selection.
Inherited replay is kept at setup; its first scheduled refresh occurs after
5,000 games in the **new stage**. A new `--weights-root` changes where this stage
saves future opponents; otherwise it inherits the parent's admission destination.

### Resume the same run

Use exact resume after an interruption or deliberate segment stop:

```powershell
python -m training.train --resume results/training/growing_run/latest.pt
```

Resume restores the saved pool, both buffers and all progress counters. It does
not scan the catalogue for extra opponents or resample merely because the
program restarted. If the saved game boundary has unfinished evaluation or
refresh work, it completes that work before another training game. To change
settings or the starting pool, create a new continuation stage instead.

### Admission and best-model rules

At the start of each evaluation block, the runner copies the current online
weights and records the current opponent pool. Both remain fixed throughout all
200 evaluation attempts. Each attempt still samples three opponents independently
from that pool. Training waits until the whole block and its processing finish.

In growing mode, a model qualifies when every evaluation attempt has an outcome
and the mean reward is **strictly greater than 0.6** by default. A game that reaches
the action cap receives reward `-1` and counts in that mean. Exactly `0.6` does not
qualify. There is no minimum number of games between admissions beyond the
evaluation schedule itself.

The exact evaluated weights are exported as a permanent neural player and added
to this run's pool. The next training game can select that opponent, and the next
evaluation uses the expanded pool. A newly admitted model is not an opponent in
the evaluation that qualified it. Another run's pool is unaffected.

New exports get a readable name containing the run ID and completed-game count.
Their permanent identity depends on weights and inference rules, so identical
models are not added twice. An existing library entry outside this pool is reused
with its original name. An already-present model can qualify again without
creating another opponent. Admission records retain the score, threshold, game
and exact evaluated pool, even when an older library entry is reused.

| Decision | Fixed pool | Growing pool |
| --- | --- | --- |
| Add a neural opponent | Never automatically | Valid evaluation mean strictly above `opponent_admission_threshold` |
| Replace `best.pt` | Highest valid mean so far; the first valid evaluation creates it | Latest valid model strictly above the admission threshold, even if an older qualifying mean was higher |
| Add evaluation games to the archive and replay | Mean strictly above `promotion_threshold` | Same rule, independent of admission |

In growing mode, `best.pt` is absent until the first qualifying evaluation. A
qualifying model can replace it even when that model is already in the pool.
Means from different pools are not directly comparable: a score of `0.61`
against a stronger pool can replace a prior `0.70`. Best-model metadata records
the opponents used to earn the score.

`promotion_threshold` defaults to `0.45`. It decides whether evaluation games
become training examples; `opponent_admission_threshold` defaults to `0.6` and
decides whether the model becomes an opponent. Changing either threshold leaves
the other unchanged. Caps lower the mean through their `-1` rewards; they do not
automatically disqualify the block.

## Refresh both buffers from the archive

`replay_refresh_interval_games` defaults to `5000` for new configurations. This
setting works in **both fixed and growing modes**. Set it to `None` in Python or
`null` in JSON to disable refreshes. Otherwise, it must be a positive whole number
of completed training episodes, including games ended by the action cap.
Evaluation games do not count.

At each refresh, the runner replaces both RAM buffers entirely. It samples
uniformly without replacement from each matching SQLite partition, up to that
buffer's capacity. If a partition has fewer records than the capacity, it selects
all of them. Only the selected records are read and checked. Invalid selections
are skipped without drawing replacements, so a buffer may contain fewer records
than its capacity. Valid selections keep archive order so future FIFO eviction
still removes the oldest first. Both replacement buffers must contain enough
valid records for their configured batches before either becomes live. The
archive itself is not changed by refreshing.

Ordinary training-game transitions live only in RAM and recovery checkpoints.
Any such records absent from the archive are lost from replay at refresh.
Qualifying evaluation games remain available because they were archived. Uniform
sampling gives older archived experience another chance to appear, but its share
still follows the archive's proportions. It does not reserve a minimum share for
RankingPlayer15 or any particular neural model.

Refresh sampling has its own saved random-number generator. It advances across
refreshes instead of restarting from the same seed. Both partitions are sampled
from one recorded archive boundary; rows appended later are eligible at a later
refresh. This also lets recovery repeat an interrupted refresh with the same
sample even if another run has since appended records.

When evaluation and refresh fall on the same game boundary, the order is:

1. Finish the training game and add its trajectory to RAM `non_demo`.
2. Evaluate; archive and promote qualifying games, then admit a qualifying model.
3. Refresh both buffers, including any examples just archived by this evaluation.
4. Save the completed boundary in the recovery checkpoint.

Recovery saves around these operations also protect intermediate progress. A
refresh-only boundary saves recovery even if the ordinary checkpoint interval is
not due. Exact resume restores completed refresh progress; continuation starts
a new local schedule without refreshing immediately.

### Archive validation

Initial replay loading and new refreshes select row IDs first, then read and
check **only the selected transitions**. With the default capacities and enough
archive records, that means 20,000 `demo` and 40,000 `non_demo` selections. The
file's format, encoder and action-map metadata, table structure and row bounds
are still checked before reading the selections.

For each selected transition, state vectors must contain 332 finite values;
masks must contain 41 booleans; the recorded action must be legal under its mask.
Rewards and terminal flags must follow the transition rules. A malformed
selection is skipped without replacement. For example, if three of the 20,000
selected demo rows fail, the replacement demo buffer contains 19,997 records.
Refresh metrics and the console summary report these skips. A refresh fails
without replacing either live buffer if too few valid records remain to supply
one configured batch from each partition. Database read errors or an incompatible
file format still stop the operation.

This checks individual examples, not every game sequence in the archive. It
does not inspect unselected payloads, compare transitions across complete games,
or recalculate all saved evaluation-block hashes. Explicit full-data reads with
`training.dataset.iter_transitions` still check every row and each game's
connecting states and terminal ending; those checks are not part of routine
replay loading. Neither kind of check replays the simulator or proves that
every otherwise valid number is historically correct.

Checks **during writing already exist**. Before appending an evaluation block,
the code validates its new complete games, archive format and existing partition
endings. It writes the new rows and tracking records in one transaction, reads
that new block back, validates it and compares its content hash before committing.
An error rolls back the entire block. The original dataset writer also checks
complete games and writes them in one transaction.

Write-time checks are unchanged. Routine replay reads now leave unselected
records alone, including any damage or outside edits that happened after those
records were written. An interrupted refresh journal created by the older code
still uses its original full-scan behavior when resumed. Newly prepared
refreshes use the selected-record checks described above.

## Train with LayerNorm: architecture version 2

Set `architecture_version=2` in a new `TrainingConfig` to place LayerNorm between
each hidden linear layer and its ReLU:

```text
332 -> Linear(256) -> LayerNorm(256) -> ReLU
    -> Linear(256) -> LayerNorm(256) -> ReLU
    -> Linear(128) -> LayerNorm(128) -> ReLU
    -> Linear(41)
```

LayerNorm rescales the hidden features separately for each observation. It uses
the same calculation during training and gameplay, with no running batch
statistics. The final layer still returns 41 Q-values. The encoder and action
mapping are unchanged, so the existing archive can supply training examples.
LayerNorm does not guarantee that Q-values stay within the possible reward range.

The ready-to-use configuration
[`layernorm_double_training_config_against_neural_1-14_from_scratch.json`](../configs/layernorm_double_training_config_against_neural_1-14_from_scratch.json)
selects architecture 2 and Double DQN. It copies the learning settings from the
existing 1–14 Double DQN configuration and enables a 1,024-observation monitoring
sample at each evaluation; the original file is left unchanged.
From `D:\Startups`, with `Startups` active:

```powershell
python -m training.train --config configs/layernorm_double_training_config_against_neural_1-14_from_scratch.json --catalog-file configs/training_pool_1-14.json --seed 42
```

This starts fresh weights in a new training directory. The selected pool can
contain architecture-1 and architecture-2 neural players together. Each opponent
is loaded using its own saved architecture; choosing version 2 for the focal
model does not convert its opponents.

To create a different configuration from an existing file, use the supported
configuration reader and writer instead of changing format numbers by hand:

```powershell
@'
import json
from dataclasses import replace
from pathlib import Path
from training.config import TrainingConfig

source = Path("configs/double_training_config_against_neural_1-14_from_scratch.json")
config = TrainingConfig.from_dict(json.loads(source.read_text(encoding="utf-8")))
config = replace(config, architecture_version=2)
with Path("configs/my_layernorm_run.json").open("x", encoding="utf-8") as stream:
    json.dump(config.to_dict(), stream, indent=2)
'@ | python -
```

`architecture_version` and `algorithm` are separate choices. Either architecture
can use `"dqn"` or `"double_dqn"`. Architecture 1 remains the default, and complete
configuration versions 1–4 load as architecture 1. Configuration version 5 adds
the explicit `architecture_version` and optional-monitoring setting
`q_monitor_sample_size`; its number describes the JSON format, not the neural
architecture.

Resume an architecture-2 run using its own full recovery checkpoint:

```powershell
python -m training.train --resume results/training/YOUR_LAYERNORM_RUN/latest.pt
```

Exact resume preserves the architecture and all saved training state. A new
continuation may change supported schedules and settings, but its
`architecture_version` must match the source checkpoint. Adding LayerNorm changes
the network's function, so a version-1 checkpoint cannot be resumed or continued
as version 2. Start a fresh run to make that change. Existing version-1
checkpoints and exported players remain readable without conversion.

## Clip training targets to the possible return range

Set `clip_targets=True` in Python, or `"clip_targets": true` in a complete
version-6 JSON configuration. This limits the **training targets** to
`[-1.5, 1.5]`, after calculating the reward plus discounted next-state value.
It works with DQN or Double DQN and either architecture. It does not change
action selection, clip the network's raw predictions, or add another loss.

For example, a calculated target of `2.9` becomes `1.5`. If the online network
predicts `2.9` for the recorded action, MSE still sees `(2.9 - 1.5) ** 2` and
pushes that prediction down. Normal terminal rewards and the timeout reward
`-1` are unchanged. A nonfinite target still rejects the update; clipping does
not conceal NaN or infinity. See the [clipping explanation](dqn_updates.md#optional-target-clipping).

The ready-to-run configuration below copies the existing LayerNorm/Double-DQN
experiment and enables only target clipping, with the required schema update:

```powershell
python -m training.train --config configs/clipped_layernorm_double_training_config_against_neural_1-14_from_scratch.json --catalog-file configs/training_pool_1-14.json --seed 42
```

To enable it in another complete config, load and export that config rather
than adding a new field to an old schema version:

```powershell
@'
import json
from dataclasses import replace
from pathlib import Path
from training.config import TrainingConfig

source = Path("configs/layernorm_double_training_config_against_neural_1-14_from_scratch.json")
config = TrainingConfig.from_dict(json.loads(source.read_text(encoding="utf-8")))
config = replace(config, clip_targets=True)
with Path("configs/my_clipped_run.json").open("x", encoding="utf-8") as stream:
    json.dump(config.to_dict(), stream, indent=2)
'@ | python -
```

Clipping defaults to **off**. Complete version-1 through version-5 configs and
checkpoints load with it off, preserving their behavior. Exact resume retains
the saved choice. To change it while keeping weights, optimizer state and replay,
use `training.continue_training --from-checkpoint ... --config ...` with the
new complete config. This is config version **6**, not architecture version 3.
Installing this change does not enable clipping in an already running process.

## Track Q-values during training

Set `q_monitor_sample_size` to a positive integer to enable monitoring just
before each evaluation's games, before any promotion or replay refresh. The
number is the **total** requested sample size, not the
number per buffer. For example, `q_monitor_sample_size=1024` requests 512 demo
and 512 non-demo observations. The default is `None` in Python or `null` in JSON,
which disables monitoring. It works with either architecture and either DQN
algorithm.

The monitor samples uniformly without replacement within each partition. If
one partition is too small, the other supplies the shortfall; if both together
are too small, it uses all available entries. An odd extra observation comes
from demo when possible. It evaluates both networks on the same observations
and masks, excluding illegal actions from the Q-value summaries.

The console prints one extra line at evaluation: the online network's best-legal-Q
median, 95th and 99th percentiles, maximum, fraction outside the return range,
and nonfinite count, plus the target network's 99th percentile. The complete
measurements are saved in the evaluation record.

Each evaluation's `q_monitor` entry in `metrics.json` contains:

| Field | What it measures |
| --- | --- |
| `online.legal_q` and `target.legal_q` | Values of every legal action across the sample. Observations with more legal actions contribute more values. |
| `online.max_legal_q` and `target.max_legal_q` | The highest legal Q-value in each observation, giving each observation one value. |
| `online.hidden_layers` and `target.hidden_layers` | Activation summaries for each hidden layer, including the distribution of positive neurons per observation and how often the whole layer is zero. |
| `td_targets` | The actual targets produced by the configured DQN or Double DQN calculation, summarized together and separately for terminal and nonterminal transitions. |
| `td_targets_unclipped` | Present when target clipping is enabled: the same target calculation before clipping. Compare this with `td_targets` to see whether raw estimates still exceed the possible return range. |
| `partition_counts`, `sample_size`, `sampling_seed` and `sample_sha256` | The actual sample composition and identifying information. |

Q-value and target distributions include minimum, maximum, mean, standard
deviation, and the 1st, 50th (median), 95th and 99th percentiles. They also report
the fractions below −1.5, above +1.5 and outside that return range. For example,
`online.max_legal_q.p99 = 4` means approximately 99% of the sampled observations
have a best legal prediction no greater than 4. Nonfinite entries are counted
separately and excluded from these statistics.

To inspect the latest saved sample in Python:

```python
import json
from pathlib import Path

metrics = json.loads(Path("results/training/YOUR_RUN/metrics.json").read_text())
latest = next(row["q_monitor"] for row in reversed(metrics["evaluations"])
              if "q_monitor" in row)
print(latest["online"]["max_legal_q"])
print(latest["target"]["max_legal_q"])
print(latest["td_targets"]["nonterminal"])
if "td_targets_unclipped" in latest:
    print(latest["td_targets_unclipped"]["nonterminal"])
```

Monitoring uses its own repeatable sampling seed and does not consume the
training, evaluation or replay random streams. It adds no optimizer updates,
does not change model modes or gradients, and does not replace replay examples.
It adds batched inference at evaluation boundaries, not to every training
update. The normal evaluation journal preserves its report across an interrupted
evaluation commit, so recovery reuses the saved measurements. When monitoring is
disabled, there is no extra sampling, inference or report.

Later evaluations sample again from the current replay, so a change in a
percentile can reflect both changed weights and changed examples. These are
replay diagnostics, not exact percentages of positions encountered in gameplay.
The reports do not clip Q-values, change admission rules or automatically stop
training. They make an emerging problem visible before inspecting checkpoints
manually.

Network Q-values are always reported **before any clipping**. With target
clipping on, monitoring adds an unclipped target pass and saves report version 2
with `clip_targets: true`. Reports without clipping retain version 1. A zero
out-of-range fraction in the clipped `td_targets` alone does not show that the
model has learned accurate values; inspect raw targets, raw Q-values and game
results too.

## Train with Double DQN

The `algorithm` setting chooses how the learner constructs its next-state
target. The default, `"dqn"`, preserves ordinary DQN. Set `"double_dqn"` to have
the online network choose the best legal next action and the target network
supply that action's value. Both train the action stored in replay. Start with
the [small numerical example](dqn_updates.md#double-dqn-in-a-small-example) if
the difference is unclear.

Both algorithms support either network architecture and use the same replay
format, player controller and training commands. The algorithm is saved in the run's
configuration, metrics and checkpoints. Existing version-1, version-2 and
version-3 configuration files load as ordinary DQN; installing this change does
not switch those experiments to Double DQN.

### Start fresh with Double DQN

Run the following from `D:\Startups` with the `Startups` environment active. It
reads the existing from-scratch configuration, changes only the algorithm, and
writes a **new** complete JSON file. The new format metadata is supplied by
`to_dict()`; the source file is not edited.

```powershell
@'
import json
from dataclasses import replace
from pathlib import Path
from training.config import TrainingConfig

source = Path("configs/training_config_against_neural_1-12_from_scratch.json")
config = TrainingConfig.from_dict(json.loads(source.read_text(encoding="utf-8")))
config = replace(config, algorithm="double_dqn")
with Path("configs/double_dqn_from_scratch.json").open("x", encoding="utf-8") as stream:
    json.dump(config.to_dict(), stream, indent=2)
'@ | python -
```

The output file must not already exist. This preserves all other settings in
the chosen source, including its learning rate, game limit and opponent-pool
growth settings. To use project defaults instead, replace the source-loading
lines with `config = TrainingConfig()` before the `replace` call.

Start a new model with that configuration and the selected opponent pool:

```powershell
python -m training.train --config configs/double_dqn_from_scratch.json --catalog-file configs/training_pool_1-12.json --output results/training/double_dqn_from_scratch
```

The output directory must not already exist. Add `--until-games 20` to save and
stop after the first 20 games without changing the full experiment's game limit.
The startup message shows `algorithm=double_dqn`, so you can check which
calculation the run will use.

### Continue existing weights with Double DQN

Continuation starts a new training directory while keeping both networks,
AdamW's stored state, replay examples and the lifetime optimizer-update count.
Changing from DQN to Double DQN does not reset the target network. The new
stage's game count starts at zero and its exploration schedule begins at
`epsilon_start`.

Use the `metrics.json` paired with the chosen parent `latest.pt` to export a
new configuration. Replace `PARENT` with that run's folder:

```powershell
@'
import json
from dataclasses import replace
from pathlib import Path
from training.config import TrainingConfig

metrics = json.loads(Path("results/training/PARENT/metrics.json").read_text(encoding="utf-8"))
config = TrainingConfig.from_dict(metrics["config"])
config = replace(config, algorithm="double_dqn")
with Path("configs/double_dqn_continuation.json").open("x", encoding="utf-8") as stream:
    json.dump(config.to_dict(), stream, indent=2)
'@ | python -
```

This retains the parent's other settings. You can also change the new stage's
game limit or epsilon schedule in `replace(...)`; see
[continuation settings](#continue-finished-training-with-new-settings).
Use the report matching the checkpoint: a newer report may describe different
settings, and after a crash the report can lag behind `latest.pt`.

```powershell
python -m training.continue_training --from-checkpoint results/training/PARENT/latest.pt --config configs/double_dqn_continuation.json --output results/training/double_dqn_continuation
```

This command inherits the parent's recorded opponent pool. Add `--catalog-file`
only when selecting a different pool for the new stage. It requires a full
recovery checkpoint such as `latest.pt`; `best.pt` does not contain the optimizer
or replay state. The parent checkpoint and reports are left in place.

### Resume a Double DQN run

Resume from the new run's own checkpoint:

```powershell
python -m training.train --resume results/training/double_dqn_from_scratch/latest.pt
```

For the continuation example, use
`results/training/double_dqn_continuation/latest.pt` instead. Resume restores the
saved algorithm and progress, including epsilon. It does not accept a new
`--config`: use continuation to change algorithms. A continuation without a
replacement configuration also inherits its source algorithm.

Double DQN makes one extra online-network prediction over nonterminal next
states per update, but still uses one backward pass and optimizer step.
It addresses a source of inflated Q-values; it does not guarantee bounded
values or recovery of inactive neurons. Keep other learning settings the same
when measuring the effect of the algorithm change.

## Configuration for future runs

Use a separate JSON configuration for each experiment. The defaults are in
[`TrainingConfig`](../training/config.py). Exporting with `to_dict()` includes
the settings and the format/version information needed to read them safely.
With `Startups` active, create an editable file from the project folder:

```powershell
@'
import json
from pathlib import Path
from training.config import TrainingConfig

config = TrainingConfig()
with Path("training_config.json").open("x", encoding="utf-8") as stream:
    json.dump(config.to_dict(), stream, indent=2)
'@ | python -
```

The file must not already exist. Open the generated JSON and change the settings
you want. For example, set `learning_rate` to `0.0001`, `training_games` to `2000`
and `epsilon_decay_games` to `2000`. Keep the other entries in the file.

```powershell
python -m training.train --config training_config.json --seed 42
```

| What you want to change | Configuration fields |
| --- | --- |
| DQN or Double DQN targets | `algorithm` (`"dqn"` or `"double_dqn"`) |
| Limit finite training targets to `[-1.5, 1.5]` | `clip_targets` (`false` by default; `true` enables clipping) |
| Hidden layers with or without LayerNorm | `architecture_version` (`1` or `2`); choose when starting fresh |
| Optional Q-value and hidden-activity reports at evaluations | `q_monitor_sample_size` (positive total sample count, or `null` to disable) |
| Optimizer settings | `learning_rate`, `weight_decay` |
| Training length and exploration | `training_games`, `epsilon_start`, `epsilon_end`, `epsilon_decay_games` |
| Target updates and discount | `target_tau`, `target_update_interval`, `gamma` |
| Number of stored examples and examples per training batch | `demo_capacity`, `non_demo_capacity`, `demo_batch_size`, `non_demo_batch_size` |
| Evaluation frequency and when its games become training examples | `evaluation_interval_games`, `evaluation_games`, `promotion_threshold` |
| Growing-pool admission and archive refresh | `opponent_pool_mode`, `opponent_admission_threshold`, `replay_refresh_interval_games` |
| How often to save recovery and when to stop an unfinished game | `checkpoint_interval_games`, `max_actions_per_game` |
| Random seed and CPU/GPU choice | `seed`, `device` |
| PyTorch CPU threads | `cpu_threads` (default `1`) |

For simple changes, no JSON file is needed:

```powershell
python -m training.train --games 2000 --seed 123 --device cuda
```

The CLI accepts `--games`, `--seed`, `--device` and `--cpu-threads` directly. To change other
settings, use the JSON file or construct `TrainingConfig(...)` in Python.
The JSON file must contain the **complete** exported configuration. A dictionary
of only the changed settings, or `dataclasses.asdict(config)` alone, is missing
required information.

New exports use configuration version 6, which adds `clip_targets`. Complete
version-1 through version-5 records retain unclipped targets. Version 5 added
`architecture_version` and `q_monitor_sample_size`. Complete version-1 through version-4 configurations
select architecture 1 with monitoring disabled.
Version 4 added `algorithm`; complete version-1, version-2 and version-3 JSON
files and checkpoints still load as
ordinary DQN. Versions 1 and 2 also use one CPU thread by default; version 3
retains its saved `cpu_threads`. Version-1 records become fixed-pool
configurations with admission threshold `0.6` and refresh **disabled**,
preserving their earlier pool and refresh behavior. Later versions retain those
saved settings.
Saving a migrated configuration writes version 6; loading does not edit the
original file. Do not change the version number by hand or add new fields to
an older-format file. Load it with
`TrainingConfig.from_dict(...)`, change settings with `dataclasses.replace`,
then export with `to_dict()` instead.

Exact resume also preserves the behavior of runs saved before automatic
admissions and refreshes were implemented, even if their configuration already
contained the new fields. Those fields were inactive in that runner, so recovery
keeps the pool fixed and refresh disabled. To enable the new behavior for such a
model, start a continuation with an explicit new configuration.

`architecture_version` selects one of the implemented networks. The exported
layer sizes, optimizer name, loss name and other format metadata describe the
supported code; changing those entries does not create a new implementation.
The configuration, architecture, observation encoder and action codec have
separate version numbers.

Each run saves the settings it actually uses inside both `metrics.json` and
`latest.pt`, including changes supplied on the command line. You can also edit
the source defaults, but that affects every later fresh run. Separate JSON files
make it easier to keep track of each experiment. A running or resumed run keeps
its learning settings; `--resume` does not accept a new `--config`. Device and
CPU thread overrides are allowed as described below.
To change settings after training has finished, use
[continuation into a new run](#continue-finished-training-with-new-settings).

### CPU threads and faster batches

`cpu_threads` is a positive whole number, defaulting to `1`. It controls
PyTorch's CPU threads for each operation, including predictions by neural
opponents on CPU. It applies to the whole Python process. It does not choose
the number of simultaneous games, Python workers or GPU threads. The separate
PyTorch inter-operation thread setting is left unchanged.

More threads do not necessarily make this small network faster. On the profiled
machine, reducing the count from 24 to 1 shortened a matched 100-game training
sample from 50.7 to 44.9 seconds and a 200-game evaluation from 28.9 to 18.3
seconds, before the array-batch improvement. Thread coordination can cost more
than it saves for small operations. Keep the default of 1 for the current GPU
training setup. Two threads were faster in a small CPU-only update benchmark,
but that has not established a faster complete CPU training run.

For example, choose two CPU threads without editing a configuration file:

```powershell
python -m training.train --config training_config.json --cpu-threads 2
python -m training.train --resume results/training/RUN/latest.pt --cpu-threads 2
python -m training.continue_training --from-checkpoint results/training/PARENT/latest.pt --config continuation_config.json --cpu-threads 2
```

The override takes priority over the JSON or saved count and is recorded in the
run's settings. It does not reset weights, optimizer state, replay or epsilon
progress on resume. Omitting it uses the configuration's count; older complete
configurations without this field use the new default of one. Existing commands
therefore work without changing their JSON files. Different thread counts may
change floating-point rounding, just as switching hardware can.

In Python, use `TrainingConfig(cpu_threads=2)` with `TrainingRun.create`.
`TrainingRun.resume(..., cpu_threads=2)` and
`create_continuation(..., cpu_threads=2)` accept explicit overrides. The runner
applies the selected setting before play. Standalone code that assembles its
own loop can call `training.runtime.configure_cpu_threads(config.cpu_threads)`
before creating or loading models.

Training now uses `buffers.sample_array_batch()` automatically. This gathers
owned NumPy arrays directly from compact replay storage, then passes them to
`learner.update`. It avoids rebuilding and checking one Python `Transition`
object for every sampled row; the learner validates array contents together.
The records drawn, RNG progression, FIFO ordering, masks and update schedule
stay the same. `sample_batch()` still provides the older tuple interface for
other callers. See [configuration and replay](neural_training_components.md#configuration-and-replay)
for the array fields and ownership rules.

## Short segments and resume

To run the first 20 games of a 10,000-game experiment, save, then continue later:

```powershell
python -m training.train --seed 42 --output results/training/first_dqn --until-games 20
python -m training.train --resume results/training/first_dqn/latest.pt
```

`--until-games` sets the total number of completed games at which to stop, not
the number of additional games. For example, resuming after 20 games with
`--until-games 40` runs another 20.

Resume writes beside the supplied checkpoint. It restores the archive path,
settings, networks, optimizer and replay buffers (past examples kept in RAM
for training). It also restores their insertion order, random-number generator
(RNG) states and progress counters. Restarting does not itself resample the
archive. Scheduled work left at the saved boundary, including a pending refresh,
is completed before the next game. The buffers use FIFO: when full, they remove
the oldest examples first.

The archive must still exist at the path saved in the checkpoint, even though
resume restores the replay buffers from the checkpoint itself.

You can change `--device` when resuming, for example to move the learner from
GPU to CPU, or use `--cpu-threads` to change PyTorch's CPU thread count without
restarting the training schedule. `--opponent-device` independently chooses the
device for restored neural opponents and defaults to CPU.
You cannot change `--games`, `--seed`, `--archive`, `--output`, `--config` or
`--catalog-file` or `--weights-root`. Resume restores the recorded pool rather than rereading its
original selection file or adding newly exported players. Referenced neural
files must still exist and match the recorded identities and fingerprints.
Keep the strategy and simulator code unchanged when continuing an experiment.
To select different opponents, start a new stage using continuation below.

On the same learner and opponent devices, recovery preserves the future random choices for training
batches and game setup, as well as the optimizer state. Results may not be
identical across different devices or software versions. Ctrl+C keeps the last
successfully saved checkpoint; work since that save may need to be repeated.
An interrupted game's examples are not added to replay. Reaching the configured
action cap is instead a scored ending, as described below. After an unexpected
error, restart from the saved checkpoint rather than reusing the in-memory run,
which may be only partly updated.

## Continue finished training with new settings

Use `training.continue_training` to start another stage from a full recovery
checkpoint, including one whose original training schedule is complete. This
creates a new run with its own output folder, run ID, game history, evaluations,
admission history, refresh schedule and best-model selection. Its `best.pt` first
appears after a valid evaluation in fixed mode, or after a valid evaluation above
the admission threshold in growing mode. Exact `training.train --resume` continues the existing run
and keeps its settings and schedule; it still does not accept configuration
changes.

From `D:\Startups`, activate the environment and continue with new settings:

```powershell
conda activate Startups
python -m training.continue_training --from-checkpoint results/training/PARENT/latest.pt --config continuation_config.json --output results/training/next_stage
```

Replace `PARENT` with the source run's folder. Use `latest.pt` or another full
recovery checkpoint; `best.pt` does not contain the optimizer and replay state
needed for continuation. The source checkpoint and reports remain unchanged.
Continuation takes the checkpoint exactly as saved and does not finish any
pending parent evaluation or refresh. If that work should be included, resume
the parent first and then continue from its updated `latest.pt`.
The new output folder must not already exist. Without `--output`, the command
creates `results/training/<UTC timestamp>_<first 12 characters of the new run ID>/`.

If `--config` is omitted, the new run inherits the source configuration. For
example, `--games 2000` starts a stage of 2,000 games using the other inherited
settings. To change several settings, export a complete configuration based on
the source run. This example reads the `metrics.json` saved alongside the selected
`latest.pt`, changes three settings, and preserves the other values and metadata:

```powershell
@'
import json
from dataclasses import replace
from pathlib import Path
from training.config import TrainingConfig

metrics = json.loads(Path("results/training/PARENT/metrics.json").read_text(encoding="utf-8"))
parent_config = TrainingConfig.from_dict(metrics["config"])
config = replace(
    parent_config,
    learning_rate=0.0001,
    training_games=2000,
    epsilon_decay_games=2000,
)
with Path("continuation_config.json").open("x", encoding="utf-8") as stream:
    json.dump(config.to_dict(), stream, indent=2)
'@ | python -
```

The JSON file must not already exist. Use the report paired with your selected
checkpoint; a later report may not describe an older checkpoint, and a crash
between file saves can leave `metrics.json` behind `latest.pt`. Continuation
checks the supplied configuration against the actual checkpoint, including
`gamma` and replay limits, but does not verify the report's identity. If no
matching report is available, omit `--config` to inherit directly from the
checkpoint; command-line overrides remain available. Keep the complete exported
configuration, including its format information.

Continuation preserves both the online and target networks, AdamW's accumulated
moments and per-parameter step counts, the lifetime optimizer-update count, and
every replay example in its original FIFO order. It does not reload or resample
the SQLite archive into RAM.
The new stage starts at zero completed training games, with its own epsilon,
evaluation, refresh and checkpoint schedules. Its first game uses `epsilon_start`.
Target-network updates still use the preserved lifetime optimizer-update count,
so their timing does not restart at zero.

You can change the `TrainingConfig` settings listed above, subject to these
requirements:

- `gamma` and `architecture_version` must match the source checkpoint.
  Continuation does not insert or remove LayerNorm. The optimizer, loss,
  observation encoding and action codec must also remain compatible.
- `algorithm` may change between `"dqn"` and `"double_dqn"` through the new
  configuration. Existing weights, optimizer state and replay are retained.
  Exact resume instead keeps the saved algorithm.
- Each replay capacity must fit all examples already in that source buffer.
  You can reduce unused capacity, but continuation cannot discard existing data
  to fit a smaller buffer. Future additions still remove the oldest examples
  when the buffer fills.
- Both batch sizes must be positive, and each must fit the retained example
  count in its corresponding buffer.
- Keep the strategy and simulator code unchanged. You may select a different
  opponent pool for the new stage as described below.

### Keep or change the opponent pool in the new stage

Without `--catalog-file`, continuation inherits the parent's recorded pool,
including neural opponents admitted during that parent run. It does not switch
back to today's default registry or automatically include newer exports.
Set `opponent_pool_mode` in the new configuration to decide whether this stage
keeps that pool fixed or continues growing it. Earlier admissions remain pool
members, but the new stage's admission history starts empty.

To change opponents while keeping learned weights, optimizer state and replay,
create another pool file with the same format as
[the training-pool example](#choose-the-opponents-for-training), then pass it to
continuation:

```powershell
python -m training.continue_training --from-checkpoint results/training/PARENT/latest.pt --catalog-file configs/next_pool.json --config continuation_config.json --output results/training/next_stage
```

`configs/next_pool.json` replaces the selected pool for this new stage; it is
not a list of additions to the parent's pool. The source run and its checkpoint
remain unchanged. The new stage's periodic evaluations use this new pool, and
its best-model score starts afresh. This avoids comparing a score against one
pool with the parent's score against a different pool.

When replacing the pool, only the neural files selected for the new pool are
needed. The parent's unused opponent files need not still exist. Inheriting the
pool or exactly resuming the parent still requires its recorded neural files.

`--opponent-device cpu`, `cuda` or `auto` chooses where the new stage's frozen
neural opponents run, including future admissions. This is separate from the
learner's `--device`. `--weights-root` chooses a new destination for future
admissions; omitting it inherits the parent's destination.

`--games`, `--seed`, `--device` and `--cpu-threads` override the matching configuration values.
`--games` sets the new stage's total; it does not also change
`epsilon_decay_games`. `--until-games 20` stops after 20 completed games in this
new stage and saves recovery, even when the parent completed thousands of games.

Without `--seed`, an unchanged configuration seed preserves the saved RNG
streams for training, evaluation, replay sampling, refreshes and global Python/PyTorch
randomness. Changing the seed in the JSON resets future randomness; changing
it to `null` uses system randomness. An explicit `--seed 42` also resets it,
even if 42 was the source seed. Reseeding does not reset weights, AdamW state
or replay data. Changed batch sizes or schedules can change later random
choices even when the saved RNG streams are preserved.

The source SQLite archive must still exist at its saved path. The new run shares
that database; it does not copy it. Future evaluation blocks that qualify for
promotion append to the shared archive and update only the current run's RAM
buffers. They do not change the source run's saved replay. The new checkpoint
and metrics include a `parent` record with the source run ID, checkpoint path,
completed-game and optimizer-update counts, and an `rng_mode` of `inherited`
or `reseeded`. Later recovery of the new run needs the shared archive, but no
longer needs the parent checkpoint.

To resume an interrupted or deliberately stopped stage later, use exact resume
on the new run's checkpoint:

```powershell
python -m training.train --resume results/training/next_stage/latest.pt
```

## Training schedule

The player being trained is called the **focal player**. Each game independently
chooses three opponents from its selected pool, so duplicate strategies are allowed.
It also randomly chooses the focal player's seat and the starting player.
Epsilon depends on the number of completed training episodes, including capped
games, and stays fixed throughout each game.

A **transition** records one focal action: the observation before it, the action,
the reward, the next observation, whether the game ended, and the legal-action
masks. The next observation is from that player's next decision or the end of
the game. Each completed transition triggers one model update using the configured
batch mix (16 `demo` and 16 `non_demo` examples by default) already in RAM. The
batch is gathered as arrays; the completed transition still signals exactly
one update. This update is run by a **callback**:
a function the episode collector calls and waits for before the player's next
choice. The final transition also triggers an update, after normal scoring or
the action cap supplies its reward and last observation, even though there is
no next choice.

The current game's sequence of transitions is its **trajectory**. The whole
trajectory enters RAM `non_demo` only after the game ends normally or reaches
its action cap, even if the player won. Ordinary training does not change `demo`.
These new training examples are saved in recovery checkpoints, not added to SQLite.

The **online network** selects actions and learns from each update. The
**target network** supplies the future values used as learning targets. With
`algorithm="dqn"`, it also selects the best legal next action. With
`algorithm="double_dqn"`, the online network selects that next action and the
target network supplies its value. Every 1,000
optimizer updates, the target weights move 10% toward the online weights
(`tau = 0.1`). This update count continues across games. Evaluation does not
advance the training-game, epsilon or optimizer counters. See
[DQN updates](dqn_updates.md) for the loss calculation and callback timing.

A game that has not finished after `max_actions_per_game` actions is a **capped
game**. The default is 10,000 actions; your configuration can set another value.
Each take or play counts as one action. A game that finishes normally on exactly
the last allowed action keeps its ordinary placement reward.

Otherwise the episode ends with reward **`-1` for every player** and no outright
winner. This is a training/evaluation reward, not a change to coin balances.
The simulator does not settle shares or reveal hidden hands at the cap.

The focal player's pending action becomes a terminal transition: reward `-1`,
`done=True`, and an all-false next mask. Its next state contains the last actual
observation, which may still show a take or play phase. DQN uses `done` to stop
bootstrapping, so it does not need a fabricated finished board. The terminal
callback runs once, and then the complete trajectory enters RAM `non_demo`.
If the focal player had not acted before a very small cap, there is no focal
transition to store or update, but the episode still has its `-1` outcome.

Capped episodes count toward the training total, epsilon schedule, evaluations,
checkpoint intervals and buffer refreshes. Training no longer stops after 100
consecutive caps. Frozen opponents do not learn from the penalty; only the focal
model receives training updates. The observation encoder remains at 332 inputs
and does not include the remaining action budget.

## Evaluation and promotion

Every 200 completed training games, the runner reports their mean reward and
evaluates a copy of the online network for 200 fresh game attempts. This copy is
**frozen**: its weights do not change during evaluation. It always chooses a
legal action with the highest Q-value (greedy selection), with no exploration.
Evaluation uses a separate random-number generator and selects opponents in
the same way as training. Its model and opponent membership are fixed at the
start of the block. Playing these evaluation games does not change the online
weights, gradients, training RNG or replay. Afterward, qualifying results can
change replay and, in growing mode, the opponent pool.

The 200 attempts form an **evaluation block**. Every attempt contributes one
reward: normal placement points if it finishes, or `-1` if it reaches the action
cap. The mean includes all 200 rewards. Caps are still counted and reported, but
they do not invalidate the block or trigger replacement attempts. A block with
caps can select a best model, admit an opponent or promote examples if its mean
meets the corresponding rule. Other errors are raised to the caller instead
of being converted into timeout rewards.

For a valid block:

- In **fixed mode**, a mean strictly above the previous best replaces `best.pt`.
  The first valid evaluation sets the best even if its mean is negative. Equal
  means keep the previous best.
- In **growing mode**, a mean strictly above `opponent_admission_threshold`
  (default `0.6`) replaces `best.pt` and admits the model if it is not already in
  the pool. This saves the latest qualifying model; it does not compare its mean
  with earlier pools' scores.
- A mean **strictly greater than 0.45** qualifies for **promotion**: adding this
  block's examples to future training. Only games the focal player won outright
  enter `demo`. Tied first place, capped games and all other outcomes enter
  `non_demo`.
- Those trajectories enter both the shared archive and the current RAM
  buffers. Both buffers remove their oldest examples when full. Evaluations
  that do not qualify add no examples.

Later evaluations use new games. Earlier evaluation games may now be training
examples, so reusing them would no longer measure performance on fresh games.

## Run files

### Output directory and layout

For a new run, `training/train.py` chooses the output folder as follows:

- Without `--output`, it creates
  `results/training/<UTC timestamp>_<first 12 characters of the run ID>/`.
  The timestamp uses `YYYYMMDDTHHMMSSZ`; the run ID is a new unique identifier
  (UUID). Separate runs with the same seed still have different folder names.
- With `--output`, it uses that exact folder instead. A relative path starts
  from the folder in which you ran the command; an absolute path is used as is.
  Parent folders are created as needed, but the run folder must not
  already exist.
- With `--resume path/to/latest.pt`, it writes beside that checkpoint and
  continues the existing run. `--output` cannot be combined with `--resume`.

When launched from `D:\Startups`, a default run has this layout. The timestamp
and ID below are examples. Files marked conditional do not always exist:

```text
D:\Startups\
|-- results\
|   `-- training\
|       `-- 20261104T153000Z_a1b2c3d4e5f6\
|           |-- latest.pt                 # Latest full recovery checkpoint
|           |-- metrics.json              # Readable configuration and progress
|           |-- best.pt                   # Conditional: passes this mode's best rule
|           |-- pending_evaluation.json   # Temporary evaluation recovery journal
|           `-- pending_refresh.json      # Temporary buffer-refresh recovery journal
|-- buffer_data\
|   `-- ranking_player_15.sqlite3         # Existing shared transition archive
`-- players\
    `-- neural\
        `-- weights\                     # Default destination for admitted models
            `-- neural_<policy ID>\
                |-- manifest.json
                `-- weights.pt
```

To choose a recognizable run name, with `Startups` active:

```powershell
python -m training.train --seed 42 --output results/training/first_dqn
```

The CLI prints the full output path after setup. Setup creates `latest.pt`
and `metrics.json`. In fixed mode, `best.pt` first appears after a valid evaluation
and changes when a later evaluation improves the best mean. In growing mode, it
appears after a valid evaluation above the admission threshold and changes at
each later qualification. The journals record pending work while its updates
are being saved and are normally removed afterward. Saving also briefly creates
`.tmp` files: the new file is written there before it replaces the old one.

Choose the archive separately with `--archive`. It is not copied into each run
folder. Evaluation blocks that qualify for promotion add their examples to
that database. Ordinary training trajectories stay in replay and its recovery
checkpoint. The settings are stored inside `metrics.json` and the checkpoints.
The output folder does not automatically get a separate configuration JSON,
files for individual games, or plots.

| File | Purpose |
| --- | --- |
| `latest.pt` | Saved training state needed to resume: both networks, AdamW state, update count, settings and format information, current opponent pool and admission destination, both replay buffers in insertion order, RNG states, recorded results, admission/refresh histories, and schedule progress. Replaced at each save. The saved archive and neural opponent paths must still exist. |
| `best.pt` | The online model selected by this pool mode's best rule, plus its settings, evaluated opponent pool and result. Can select actions but cannot resume optimizer updates. Absent until an evaluation passes that rule. |
| `metrics.json` | Readable settings, current opponents and results: replay sizes; game rewards, epsilon, action/transition counts and loss; evaluation scores, caps, evaluated opponents, promotion, admission and best status; admission and refresh histories; optional `q_monitor` reports inside evaluation records. Updated with recovery saves and normal completion. |
| `pending_evaluation.json` | Temporary record of evaluation results being saved. Normally removed after the following recovery save; kept if that process is interrupted. |
| `pending_refresh.json` | Temporary refresh record, including its sampling seed and archive row boundaries. Allows an interrupted refresh to produce the same replacement buffers on resume. |

Reward means points for final placement: 1.5, 0.5, -0.5, -1.5 for first through
fourth, with ties averaging the occupied places. A capped game instead gives
everyone `-1`, so that outcome is not zero-sum. Reward is separate from the final
coin balance. Mean loss is the average loss across that game's DQN updates.
Training and evaluation results are recorded separately.

`buffer_sizes` records how many transitions are currently stored in `demo` and
`non_demo`, not their maximum capacities. Each game's sizes are recorded after
adding its transitions and before evaluation. Evaluation sizes are recorded
after any promotion. Capped games have ordinary game records, including their
`-1` reward, `capped=True`, buffer sizes and loss. New records identify their
`reward_scheme`; evaluation records also include a `capped_mask` in game order,
so a timeout can be distinguished from a normal result with reward `-1`.
Older discarded-attempt records stay as saved.
If older records lack newer fields, resuming does not invent earlier values.

For growing pools and refreshes, these fields help explain what happened:

- Top-level `opponents` is the current pool. Each evaluation's
  `evaluated_opponents` is the pool used for that block, before any new admission.
  `pool_size_after` records the size after processing it.
- An evaluation's `admission` records whether it qualified, whether it added a
  member, the reason, and the model ID and path when applicable. A duplicate can
  qualify without being added again.
- Top-level `admissions` records actual joins, with their score, threshold,
  completed-game count and evaluated pool. It includes joins that reused an
  existing library entry.
- The existing `improved_best` field means that `best.pt` was replaced. In growing
  mode this can be true even when the new mean is lower than the previous one.
  Top-level `best_mean` and `best_game` describe the model currently in that file.
- Top-level `refreshes` records each refresh's game, sampling seed, archive
  boundary, and buffer sizes before and after replacement. New records also
  include `skipped_invalid`, the number of selected invalid rows skipped in each
  partition. `archive_snapshot` identifies the archive and each partition's
  `row_count` and `max_row_id`.

The archive boundary and seed explain which data was eligible and how the sample
was chosen. The checkpoint itself holds the resulting buffers and their RNG
states, so ordinary resume does not need to rebuild a completed refresh.

The CLI prints a training summary when evaluation is due, the evaluation result,
admission and refresh information, and a summary when the run completes or reaches
a requested segment stop. It does not print every game. If you change the evaluation interval, the training
summaries use that same interval.

Recovery is saved at setup, after the scheduled number of games regardless of
model performance, around evaluation and refresh processing, and at normal
completion or a segment stop. A 10,000-game run has 50 scheduled saves at
200-game intervals. Evaluation recovery may require several saves at the same
game count, but **only one latest recovery file is kept**, alongside the separate
best model. Each save writes a temporary file first and then replaces the old
file in one operation. If writing the new file fails, the old file remains.

The checkpoint gets larger as the replay buffers fill up. Full buffers at the
default capacities contain about 157 MiB of numeric data. The complete PyTorch
recovery file is larger because it includes other state too. Saving and loading
also need temporary memory. Replay is stored in a compact numeric format rather
than as full Python `Transition` objects.

## Safe archive updates and recovery

Transition format 2 is unchanged. Promotion adds two database tables,
`evaluation_blocks` and `evaluation_episodes`, to track what has already been
saved. These tracking tables use ledger version 1. Each block is identified by
its run ID and completed training-game count. Its record includes hashes
(fingerprints) of the model and contents, plus the RNG state after evaluation.
The example rows and tracking records are saved in one SQLite transaction:
either all of them are committed or none are.

If an interrupted save is retried with the same contents, it reuses the saved
block without adding duplicates. A retry with conflicting contents is rejected.
Each append checks the new games and the ends of the existing `demo` and
`non_demo` groups, without rereading the whole archive. Normal dataset loading
still reads only `demo` and `non_demo`.

Suppose the database save succeeds, but the process stops before the recovery
checkpoint is updated. On resume, the runner finds the saved evaluation,
restores its ending RNG state and adds its examples to RAM once. The temporary
JSON journal also keeps results from evaluations that did not qualify, before
the best model and recovery files are saved. These records prevent duplicate
database rows, duplicate additions to RAM and unnecessary repeated evaluations
after an interrupted save.

Growing-pool evaluation records also identify the exact evaluated opponent pool.
The same frozen candidate supplies both its permanent export and `best.pt`.
If exporting succeeds before recovery is saved, retrying reuses its permanent
identity instead of creating another model. The saved pool and admission history
ensure the policy joins once. A library file left by an interrupted export can
already appear in the catalogue; it becomes part of the resumed run's pool when
that admission finishes. Other running pools do not change.

Refreshes use `pending_refresh.json` to preserve their sampling seed and archive
row boundaries before replacing live buffers. Both replacement buffers are built
and checked first. If the process stops before their recovery save completes,
resume repeats the sample from the recorded boundary, even if the shared archive
has since grown. If the save completed but journal cleanup did not, the recorded
refresh progress prevents another replacement. The archive is append-only during
normal use; do not edit or delete old transition rows between recovery attempts.
The journal also records the refresh format: older pending refreshes retain
their full-scan checks, while new ones check only their selected records.

The timeout rule is recorded as reward scheme
`placement_points_with_timeout_v2`. New training outcomes and evaluation blocks
use it. Recovery records where that scheme began, using
`reward_scheme_start_game`; `reward_scheme` identifies the current rule.
Loading an older checkpoint keeps its learned weights, optimizer, replay and
epsilon progress, while future games use the `-1` timeout rule. Existing metrics
and replay rows are not rewritten. If an old evaluation was already saved in a
pending journal or archive block, recovery finishes that block under its original
rules before starting new evaluations. This prevents a partially committed result
from changing meaning during recovery.

A process already running when the code changes keeps its loaded behavior.
Stop it and use the normal `training.train --resume .../latest.pt` command with
the updated code to use the new timeout rule. Epsilon continues from that saved
checkpoint's completed-game count; it does not restart at `epsilon_start`.

New datasets use the new reward-scheme header. Readers also accept the old
`placement_points_v1` header. The first append of new-scheme evaluation data
upgrades an old archive's header transactionally without changing historical
rows. Older program versions do not understand that new header; use the updated
code when reading the shared archive. The encoder stays at 332 inputs and the
network architecture is unchanged, so existing neural weights remain usable.

Older runs still restore their original fixed-pool behavior
with refreshes disabled. Their recovery state is upgraded when saved again.
Starting a continuation with an explicit new configuration is how to enable the
new features for a previously trained model.

## Modules and tests

The original seven steps map to the implementation as follows:

| Step | Implementation |
| --- | --- |
| 1. Network and controller | `players/neural/q_network.py`, `players/neural/base.py` |
| 2. Configuration and replay | `training/config.py`, `training/runtime.py`, `training/replay.py`, `training/batches.py` |
| 3. DQN update | `training/dqn.py`: `DQNLearner.update` |
| 4. Episode callbacks | `training/episodes.py`, `training/transitions.py` |
| 5. Evaluation and archive updates | `training/evaluation.py`, `training/archive.py` |
| 6. Checkpoint saving and recovery | `training/checkpoints.py` |
| 7. Entry point, reporting and documentation | `training/train.py` and these guides |

| Module | Responsibility |
| --- | --- |
| `training/evaluation.py` | Evaluate frozen weights against one recorded pool without exploration; decide whether its games qualify for promotion and which buffer receives each game. |
| `training/q_monitor.py` | Optionally sample replay independently and summarize online/target Q-values, training targets and hidden-layer activity at evaluation boundaries. |
| `training/opponent_pool.py` | Apply admission rules, reuse permanent neural identities, append new pool members and record when they joined. |
| `training/archive.py` | Save whole evaluation blocks without partial or duplicate entries; retrieve saved blocks after interruption. |
| `training/checkpoints.py` | Check, save and load recovery/best files; restore RNG states. File replacement happens only after the new file is fully written. |
| `training/replay.py` | Load and sample examples, remove the oldest when full, and save buffers in a compact format. |
| `training/archive_sampling.py` | Select archive row IDs, read and check only those examples, and skip invalid selections without replacements. |
| `training/batches.py` | Own and validate numerical training batches without rebuilding individual transition objects. |
| `training/runtime.py` | Apply the selected process-wide PyTorch CPU thread count. |
| `training/replay_refresh.py` | Record a consistent archive boundary, sample both replacement buffers and validate them before use. Reads the archive without changing it. |
| `training/train.py` | Run the schedule, record and print results, handle recovery, and provide the CLI. |
| `training/continue_training.py` | Start a new stage from a full recovery checkpoint with compatible configuration changes. |
| `players/catalogue.py` | List available models, load selected pools, create their players, and restore recorded models for recovery. |

```powershell
python -B -m unittest discover -s tests
```

Tests cover CPU/CUDA recovery, matching random samples and AdamW updates after
resume, comparisons with uninterrupted runs, action caps, the strict promotion
threshold, separation of `demo` and `non_demo`, and preservation of the original
archive. They also simulate failures while saving the archive, best model and
recovery file. Short GPU command-line checks cover starting and resuming with
separate output folders, without adding examples to the original archive.
Growing-pool and refresh tests cover strict admission thresholds, duplicates,
pool changes after evaluation, latest-qualified best selection, archive sampling,
continuation and recovery across interrupted boundary operations. Tests use
temporary archives and model libraries.
