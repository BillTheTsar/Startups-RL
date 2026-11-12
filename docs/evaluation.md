# Evaluating automated players

Use `evaluate.py` to compare automated players. It runs games through
`game_loop.run_game`. Every game has **exactly four players**, chosen from a
pool of **at least four distinct model IDs**. A model ID names a configuration,
not a Python class. For example, ranking profiles are evaluated separately even
though they all use `WeightedRankingPlayer`.

```powershell
python evaluate.py --list-models
python evaluate.py --games 1000 --seed 42
python evaluate.py --games 1000 --seed 42 --models RandomPlayer ThresholdPlayer1 RankingPlayer1 RankingPlayer3
python evaluate.py --games 1000 --seed 42 --output-root D:\EvaluationResults
python evaluate.py --rebuild-report "results/evaluations/RUN_ID"
```

With no catalogue options or `--models`, the pool contains all built-in
rule-based models. Use `--list-models` to see their IDs. Ranking configurations have IDs such as
`RankingPlayer1`; threshold configurations use IDs such as `ThresholdPlayer1`
or `ThresholdChicken1`. The command rejects unknown IDs, repeated IDs, and pools
with fewer than four models. `--games` is the **total game count**, not a count
per model or pair.

## Evaluate a selected mixed pool

An exported neural player can be evaluated alongside the rule-based players.
There are two choices to make: which models are available on disk, and which
of them belong to this evaluation's pool. You do not need a different Python
catalogue script for each experiment.

First export a trained checkpoint, if you have not already done so:

```powershell
conda activate Startups
python -m players.neural.export_model --checkpoint results/training/YOUR_RUN/best.pt --name "My neural opponent"
python -m players.catalogue --list-models
```

Replace `YOUR_RUN` with the source run's folder. The second command lists all
built-in rule-based IDs and exported neural IDs. Copy the neural player's full
`neural_...` ID. Its readable name is a label, not the value used to select it.

Create `configs/evaluation_pool.json` containing your chosen IDs. This example
has four entries, the minimum for standalone evaluation; replace the placeholder
with the actual full neural ID:

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

The file must contain unique IDs. Its order becomes the initial model order for
the evaluation, so reordering it changes a seeded run. Neural IDs normally refer
to `players/neural/weights/`. To use another exported-model library, add a
`weights_root` field. A relative root is resolved from the JSON file's folder;
for example, `"weights_root": "../saved_neural_players"` in this file selects
the project's `saved_neural_players/` folder.

Inspect the selected pool, then run the evaluation:

```powershell
python evaluate.py --catalog-file configs/evaluation_pool.json --list-models
python evaluate.py --catalog-file configs/evaluation_pool.json --games 1000 --seed 42
```

Each game samples **four distinct IDs** from the selected pool. With exactly
four entries, all four play every game, in randomly assigned seats. With more
entries, only four appear in each game. This differs from human play and DQN
training, where three opponents are sampled with replacement and duplicates
are allowed.

Omitting `--models` uses every entry in the file. Adding `--models ID1 ID2 ...`
selects a further subset, which must still contain at least four distinct IDs.
The `--list-models` command lists the catalogue selected by `--catalog-file`;
`python -m players.catalogue --list-models` lists the larger available library.

Neural opponents use CPU by default. Add `--opponent-device cuda` or
`--opponent-device auto` to change where their fixed networks run:

```powershell
python evaluate.py --catalog-file configs/evaluation_pool.json --games 1000 --seed 42 --opponent-device cuda
```

The selected networks are loaded once and remain unchanged. Every game creates
fresh controllers for its seats. Adding exports or editing the pool file while
the command is running does not change that run's membership. Its saved manifest
records the selected identities and neural weight fingerprints.

The existing `--catalog module:attribute` interface for custom Python factories
still works. Choose either `--catalog` or `--catalog-file` for a command; they
cannot be combined. Report rebuilding needs neither the pool file nor the
exported weights and does not load neural networks.

If a selected combination keeps playing without finishing, the action cap ends
that game with **`-1` point for each player** and evaluation proceeds to the next
game. No player is forced to choose a different action to escape a cycle.

## Module responsibilities

- `evaluate.py` is the command-line entry point. It selects models, starts runs,
  connects the functions that save results, and handles progress, errors, and
  report rebuilding.
- `evaluation_reports.py` formats console tables and generates JSON, CSV, and
  text exports from saved results.
- `evaluation_storage.py` saves run descriptions (manifests) and completed-game
  records in a SQLite database. It saves batches periodically and reads saved
  runs to recalculate statistics.
- `evaluation.py` chooses players, manages random seeds, scores games, and combines
  game results into statistics. It runs each game through `game_loop.run_game`
  and provides callbacks: functions the caller can supply to save results.
- `players/catalogue.py` lists available models and loads the selected pool.
  `ModelSpec` remains importable from `evaluation` for existing Python catalogues.

The gameplay call chain is `evaluate.py` -> `evaluation.evaluate()` ->
`game_loop.run_game()`. Rebuilding reports reads saved outcomes without running games.

## Sampling and reproducibility

For each game, four different models are randomly assigned to seats P1..P4.
Every possible ordered lineup is equally likely; a model cannot appear twice
in one game. The starting seat is chosen separately, with each seat equally
likely to start. Turns then advance P1 -> P2 -> P3 -> P4 -> P1.

The evaluator does not balance appearances, adjust the schedule, or adjust
probabilities for model families. Every model in the selected pool has the same
chance of appearing, even if many models share one class. As a result, appearance
counts, games between each pair, and starting positions will naturally vary.

`--seed` controls the run's random choices. If you omit it, the evaluator
generates a seed and prints it before play. A random number generator (RNG) used
only for this run creates a separate seed for each game. Each game has its own
RNG, which chooses the lineup and starter and creates separate seeds for the
deck and each player. The evaluator does not reset any process-wide RNG.

Every game creates a new simulator and new player controllers. The game loop
resets each controller once without resetting its random seed. Policies that
always make the same choice for the same input may ignore their seed.

To reproduce a run, use the same models in the same order, the same source code
and configurations, and the same Python RNG behavior. Custom model adapters
must also behave the same way for the same inputs and seeds. Neural frameworks
may have their own RNGs or operations whose results vary between runs; their
factories/adapters must manage these. Do not train models or update weights
during evaluation. Reordering the model pool changes a seeded run.

## Points, wins, and placements

Players are placed by their final simulator coin balances, which may be
negative. Evaluation points are separate from coins:

| Place | Points |
|---|---:|
| First | +1.5 |
| Second | +0.5 |
| Third | -0.5 |
| Fourth | -1.5 |

Tied players share the points for the places they occupy, taking the average.
Two tied first-place players each get +1; two tied second-place players each
get 0; four tied players each get 0. The points in each normally settled game
sum to zero.

A game that reaches its action cap instead gives **`-1` to all four players**,
for a total of `-4`. It has no outright winner and no ranked placement. This is
an explicit timeout outcome; no final coin settlement or hidden-hand reveal is
invented. It is included in appearances and mean points, so repeated timeouts
lower a model's leaderboard score.

The leaderboard shows total points but sorts by **mean points per appearance**
(total points divided by games played), because models may play different numbers
of games. An **outright win** means finishing alone with the highest coin balance.
Outright win rate is outright wins divided by all the model's completed
appearances, including games with ties and timeouts.

Placement distributions show how often a model finishes in each place. Ties
split credit across the occupied places: two players tied for first each
contribute 1/2 to first and 1/2 to second. Three players tied for second each
contribute 1/3 to second, third, and fourth. These fractions are added exactly,
then converted to floating-point numbers for reports. Capped games earn no
placement credit. Thus the four placement percentages can sum to less than
100%; the missing share consists of capped appearances. First-place credit can
still exceed outright win rate because it includes tied first places.

The report also shows appearances, mean points, outright wins/rate, and placement
distribution separately for positions 1..4 **relative to the starter**. These
positions are turn order, not fixed seat IDs. If P3 starts, P3 is position 1,
P4 position 2, P1 position 3, and P2 position 4. Each position's means and rates
use only appearances in that position. A model or position with no appearances
shows `N/A` rates, not zero performance.

## Normalized head-to-head matrix

The matrix compares every pair of models that finished a normally settled game
together. Capped games have no ranked result and are excluded from this matrix.
Finishing with more coins than the other player counts as a win over that player, regardless
of who won the whole game.

```text
matrix[A, B] = (A's wins over B - A's losses to B) / their shared settled games
```

Ties add zero to wins minus losses, but still count as one shared settled game.
A result of `+0.30` means the win rate exceeds the loss rate by 30 percentage points; it
does not mean a 30% win rate. Positive values favor the row model. Swapping the
row and column reverses the sign. The diagonal (a model compared with itself)
is zero by convention. Pairs without a shared settled game are `None` in the API
and `N/A` in the report, even if they appeared together in capped games. A pair
whose shared settled games were all ties has value zero.

Only this normalized matrix is returned and displayed. The evaluator keeps
counts and wins-minus-losses totals internally to combine results correctly;
it does not output them as extra matrices. Matrix rows and columns use the
model order in the manifest, regardless of leaderboard sorting. The printed
model key explains short labels such as M1 and M2. These results describe games
drawn from the selected pool: the other two opponents in a four-player game can
affect how a pair performs against each other.

## Saved runs and rebuilding reports

Every command-line evaluation creates a new, uniquely named folder under the
project's `results/evaluations/`, regardless of the current working directory.
Use `--output-root PATH` to choose another parent folder. A relative custom path
starts from the working directory. Folder names contain a UTC timestamp and a
random suffix. Existing runs are never overwritten, resumed, or extended.

```text
results/evaluations/<run-id>/
    run.sqlite3
    manifest.json
    summary.json
    overall.csv
    by_position.csv
    head_to_head.csv
    report.txt
```

**`run.sqlite3` is the source of truth for saved results.** It stores the original
manifest, each game record that has been committed (saved in a completed database
transaction), and details about the run: run ID, timestamps, saved game count,
save interval, final status, and any recorded error. All six other files are
generated from this database and can be recreated. The evaluator uses Python's
standard-library `sqlite3` module; no database server is needed.

The manifest describes how the run was set up. It includes the actual master
seed, requested game count, four-player assumption, rules for choosing players,
scoring and ties, action limit, model IDs/kinds and configurations, Python and
action codec versions, and SHA-256 hashes (file fingerprints) of the local
evaluator, simulator, and player Python sources. Built-in model metadata stores
the full weights/thresholds as fixed values. Fraction weights are stored as
numerator/denominator objects. Neural players selected through `--catalog-file`
record their permanent identity, exported-model location and weight fingerprint.
External custom model source files and checkpoint contents supplied through
`--catalog module:attribute` are not automatically fingerprinted; include their
versions/hashes in custom model metadata.

The database contains everything needed to recreate **the current evaluation
reports**. It does not record complete gameplay: individual actions, intermediate
states, model checkpoint files, and source file contents are not saved. Seeds
and source hashes identify a run but do not replace those files.

To rebuild exports, the evaluator reads saved final scores, determines placements
and ties, and recalculates points, outright wins, placement distributions,
starting-position results, and the normalized matrix. It does **not** draw cards,
choose actions, run the simulator, or import player implementations or custom
catalogs.

```powershell
python evaluate.py --rebuild-report "results/evaluations/RUN_ID"
```

Run this after an evaluation has stopped if reports were deleted, an export
failed, or the process exited before creating them. It can also recreate
`manifest.json`. Rebuilding uses the saved model identities/configurations and
supported saved scoring rules. You can read an old archive even if model weights
have changed, custom modules are unavailable, or checkpoints are missing. The
rebuild command explicitly rejects unknown storage/evaluation versions and
inconsistent scoring rules. It cannot be combined with game/model/seed/catalog
selection.

Each game record contains the game number and seed, the model in each seat,
starting seat, deck/player seeds, four coin balances, action count, and a
`capped` flag. For a normal ending, those balances are final settled scores.
For a timeout, they are the public balances when play stopped, and the scorer
uses `-1` points for everyone instead of ranking those balances.
Records use JSON inside SQLite to preserve 64-bit unsigned seeds exactly; these
can exceed SQLite's signed-integer range. The game number uniquely identifies
the database record. Reading checks that game numbers are consecutive and that
saved counts match.

New manifests use evaluation version 2 and record the timeout reward, absence of
timeout placement credit, and exclusion of timeouts from head-to-head results.
The SQLite storage schema remains version 1. Old version-1 manifests and records
without a `capped` field still rebuild as originally scored; old reports are not
reinterpreted as if timeout outcomes had been saved.

Reports show the total number of games ended at the cap and a `Capped` count for
each model. Summary JSON contains top-level `capped_games`; model and position
statistics include `capped_games` and `capped_rate`. CSV exports include the same
counts and rates. Their denominators include all appearances, so timeout rates,
win rates and placement distributions describe the same set of games.

## Incremental saving and recovery

The manifest is committed before the first game. Completed outcomes wait in a
queue for a background writer, which commits a batch **every 30 seconds** using
its own timer and SQLite connection. The timer does not wait for another game
to finish or for the game's asyncio loop to give other work a turn. Normal
completion, Ctrl+C, and handled game errors save the remaining completed outcomes
and the final status. A game interrupted by an error or Ctrl+C has no result.
A game ended by the configured action cap does have its `-1` timeout outcome.

Each batch and its saved-game count are saved in one transaction, so they succeed
or fail together. SQLite uses rollback journaling (a recovery file for interrupted
writes) and `synchronous=FULL`. After abrupt termination or power loss, SQLite
recovers the last committed state. Normally, up to 30 seconds of results are
waiting to be saved. Scheduling delays, slow or stalled storage, or a suspended
machine can lengthen this window. Unsaved outcomes and the unfinished game cannot
be recovered in this version. There is no resume feature.

After a crash, a temporary `run.sqlite3-journal` file may remain. Keep it beside
the database: SQLite may need it to undo an interrupted transaction when opening
the archive. After a clean close or recovery, `run.sqlite3` contains the complete
archive. When in doubt, keep or copy the whole run folder. Avoid backing up a
database by copying it while it is being written.

Database errors stop evaluation at the next game/action check. This check cannot
interrupt a model call that is currently blocked. The command reports the problem,
tries to finish saving and closing the run, and exports only committed records
it can read. It never presents unsaved results held in memory as saved results.

Saved final statuses are `completed`, `interrupted`, and `failed`. If no final
status was committed, exports say `unfinished`. This can mean the run crashed
or that the export captured a run still in progress. Reports show both saved
and requested game counts. Rebuilding reports never fills in missing games or
changes the run status.

Exports are generated when the run completes or handles a stop, and when you
use `--rebuild-report`. Periodic commits do not generate exports. Each export is
written to a temporary file and flushed before replacing the destination file
with the completed version. If an export is locked (for example, open in Excel)
or interrupted, the database remains the source of truth; close the file and
rebuild. The six exports are not replaced as one indivisible operation. Rebuild
again if files contain results from different points in the run.

## Export formats

- `manifest.json`: the original manifest, also stored in SQLite.
- `summary.json`: manifest, run status, overall statistics, statistics by relative
  position, and one normalized matrix with its ordered model IDs. Includes
  calculated means/rates.
- `overall.csv`: one model per row, sorted by mean points. Includes appearances,
  total/mean points, outright wins/rate, fractional placement counts and rates.
- `by_position.csv`: the same statistics per model and relative position 1..4.
- `head_to_head.csv`: only the normalized matrix, labeled with full model IDs in
  manifest order. The accompanying summary/report identifies the run and saved
  results used for this export.
- `report.txt`: the readable console tables plus run status, saved game count,
  and any recorded error.

CSV rates are numbers from **0..1**, while the text report displays percentages.
JSON uses `null` and CSV uses `N/A` for unavailable rates or unplayed pairs.
The matrix diagonal is zero. The overall and positional CSVs also include run
ID, status, saved-game count, and requested-game count. No matrices of counts or
wins-minus-losses totals are exported.

## Python API

`evaluation.evaluate` returns an `EvaluationResult` containing the manifest,
completed game count, overall model statistics, model statistics by relative
position, and the normalized matrix. Direct Python calls do not write files
unless you supply storage callbacks. The command-line interface always supplies
them.

```python
import asyncio
from evaluation import builtin_models, evaluate

result = asyncio.run(evaluate(builtin_models(), 1000, seed=42))
for model in result.models:
    print(model.model_id, model.overall.mean_points, model.overall.outright_wins)
```

You can supply synchronous callbacks (ordinary functions, not async functions):

- `on_start(manifest)` runs before any game is constructed.
- `on_game(record)` receives each completed game after its result has been added
  to the statistics.
- `check_status()` runs before each game and after each action.

The command-line interface uses these callbacks to initialize `RunArchive`,
queue outcomes with `append`, and check background errors with `check`. It calls
`finish` during cleanup.

`dataclasses.asdict(record)` produces data that can be saved as JSON. Pass compact
records to `EvaluationAccumulator` to recalculate all current statistics. The
runner does not keep these records or full histories across games. Memory for
combined statistics is O(m squared), where `m` is the number of models, because
it includes pairwise results. The archive holds records in memory only until
its next save. `evaluation_storage.load_run` returns a `SavedRun`.
`evaluation_reports.rebuild_reports` reads the archive and recreates its exports.
Reading the archive does not require the separate report files.

## Custom and neural models

`ModelSpec` accepts any implemented `Player` factory: a function that creates a
player, with signature `factory(player_id, *, seed)`. The returned player must
have the requested seat and declared kind (`rule_based` or `neural`). Human
controllers are excluded. Controllers can share neural weights if those weights
cannot change. Each controller must have its own RNGs and game-specific state,
including any recurrent state carried between decisions.

To select custom models from the command line, define a sequence of specs in a
Python module that can be imported, then use `--catalog module:attribute`. This
replaces the built-in catalog. Include built-in specs explicitly if you want
to mix them with custom models:

```python
# my_models.py
from evaluation import ModelSpec, builtin_models
from my_policy import create_player

MODELS = (*builtin_models(), ModelSpec(
    "checkpoint_100_greedy", create_player, kind="neural",
    config={"checkpoint_id": "checkpoint_100", "selection_mode": "greedy"},
))
```

```powershell
python evaluate.py --catalog my_models:MODELS --games 1000 --seed 42 --models checkpoint_100_greedy RandomPlayer ThresholdPlayer1 RankingPlayer1
```

`NeuralPlayer` is implemented in `players/neural/base.py`. It can use
`FlatObservationEncoder()` with a loaded `QNetwork.predict` adapter. The loader
`training.checkpoints.load_best` loads a saved inference model for evaluation;
see [training](training.md) for checkpoint details. A custom catalogue can use
that loader in its factories. For the supported exported-player format, you do
not need to write factories yourself: use the
[JSON pool workflow](#evaluate-a-selected-mixed-pool) instead.

## Progress, errors, and limits

Progress is printed to stderr (the error/progress output stream) every 100
completed games and at completion. Change the interval with `--progress-every N`,
or disable progress with `--progress-every 0`.

The default `--max-actions 10000` limit counts individual take/play decisions,
not complete player turns. A game that finishes normally on exactly the limiting
action keeps its ordinary placement points. Otherwise, that game ends with `-1`
for every player, is recorded as capped, and evaluation continues. This limits
the number of actions; it does not limit how long a model can take to choose an
action.

A game error stops evaluation immediately. The command records
the error and complete setup of the failing game, saves completed games, then
prints/exports saved statistics and exits with status 1. Failed games are never
scored as draws or replaced with newly sampled games. In Python,
`EvaluationGameError` exposes `plan` and `partial_result`, with the original
exception as its cause.

Invalid command-line arguments exit with status 2. Ctrl+C cancels the run, saves
completed records, exports partial statistics, and exits with status 130.
Storage/export failures exit with status 1. Repeated interruption during cleanup
can prevent the final save; previously committed batches remain available.
