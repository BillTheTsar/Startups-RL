# Save and play against a neural player

[Documentation index](README.md)

Exporting saves a permanent copy of a trained model so that it can be used as a
player later. Your training run can continue replacing its `best.pt` and
`latest.pt`; the exported player keeps the weights it had when you saved it.

After exporting, choose the players you want in a small JSON pool file. The
human-play, evaluation and training commands can use that file. Exporting a
player does not automatically add it to any opponent pool.

## Export command

Run this from the project folder (`D:\Startups`), replacing `YOUR_RUN` with the
folder of the training run you want to use:

```powershell
conda activate Startups
python -m players.neural.export_model --checkpoint results/training/YOUR_RUN/best.pt --name "My neural opponent"
```

The command prints the saved player's identity and folder. `--name` supplies a
readable label; it is optional. The source must be a supported checkpoint file,
not the folder containing it.

You can also export the model from a recovery checkpoint:

```powershell
python -m players.neural.export_model --checkpoint results/training/YOUR_RUN/latest.pt --name "Latest training model"
```

| Source | Which weights become the player? |
| --- | --- |
| `best.pt` | The online network saved when that checkpoint was selected as best. |
| `latest.pt` | The online network saved at that recovery point. The target network is not used. |

The exporter reads the source without changing it. It does not start or resume
training, restore an optimizer or replay buffers, or change the caller's random
number streams. Reading `latest.pt` still involves reading its stored recovery
data into memory, so exporting it can use more memory than exporting `best.pt`.

Exports go under `players/neural/weights/` by default. This default is relative
to the project itself. To choose another location:

```powershell
python -m players.neural.export_model --checkpoint results/training/YOUR_RUN/best.pt --output-root saved_neural_players --name "My neural opponent"
```

The output root is a folder that can contain many saved players. The exporter
chooses the individual player's folder name from its identity.

## What is saved in each entry?

Each exported player has its own folder containing two files:

```text
players/neural/weights/
  neural_<hash>/
    manifest.json
    weights.pt
  neural_<another_hash>/
    manifest.json
    weights.pt
```

The actual hash is a full 64-character SHA-256 value. It is a fingerprint
calculated from the weights and the rules for interpreting those weights.

`manifest.json` is a readable description of the player:

| Information | Why it is stored |
| --- | --- |
| Permanent model identity and readable name | Identify the exact player and display a useful label. |
| File-format version and creation time | Describe how the entry is stored and when it was first saved. |
| Network architecture and numeric type | Rebuild the network with the correct layers and weight precision. |
| Observation encoder | Identify the expected 332-value input format. |
| Action mapping | Identify what each of the 41 output positions means. |
| Selection rule | Specify how Q-values and the legal-action mask produce a choice. |
| Weight filename and file fingerprint | Locate the weights and detect a changed or damaged file. |
| Source information | Record the source checkpoint and available training or evaluation details. |

`weights.pt` contains the learned network parameters: the weights and biases of
its layers. It has no replay examples, optimizer state, target network, or
training counters. Together with `manifest.json`, it supplies what is needed to
recreate the player. Loading it no longer requires the original checkpoint or
the training archive.

Both supported architectures have sizes `332 -> 256 -> 256 -> 128 -> 41`.
Version 1 uses a ReLU after each hidden linear layer. Version 2 adds LayerNorm
between each of those linear layers and its ReLU. Its saved parameters include
LayerNorm's learned scales and biases, and the manifest records the normalization
settings. Both versions keep a final linear layer.

The export command is the same for either architecture: it reads the version
from the checkpoint. Existing version-1 entries load unchanged and retain their
permanent IDs. A selected pool may contain both versions together; each model
is rebuilt from its own manifest. No conversion of older players is needed.

Exported players use the current observation encoder and
action mapping. They choose the legal action with the highest Q-value; a tie
chooses the lowest action ID. They do not take random exploratory actions.

## Permanent identities and repeated exports

The identity depends on the exact saved weights and the inference settings,
including the architecture, encoder, action mapping and selection rule. It does
not depend on the readable name, export time or source checkpoint path.

Exporting the same weights with the same inference settings into the same output
root therefore reuses the existing entry. It keeps that entry's original name
and source information. Giving the export a different name does not create a
second opponent with the same weights.

Different weights produce a different identity and folder. Existing entries are
checked before reuse; an incompatible or damaged entry is reported rather than
overwritten. New entries are published only after both files have been written.

## Load an exported player in Python

Use the folder printed by the export command:

```python
from players.neural.artifacts import load_neural_artifact

saved = load_neural_artifact("players/neural/weights/neural_<hash>", device="cpu")
player = saved.create_player(1)
```

Replace `neural_<hash>` with the actual folder name. The loader checks that the
files are intact and compatible with the current code. The loaded network is in
evaluation mode with gradient calculation disabled for its parameters. Loading
preserves the caller's random number streams.

`player` is an ordinary `NeuralPlayer` for seat 1. The existing game loop can call
its `choose_action` method. You can create separate controllers for other seats
from the same loaded model; they share its fixed weights and have their own
player identities and controller state.

Loading defaults to CPU. `device="cuda"` requests the GPU; `device="auto"`
uses the GPU when available and otherwise uses CPU.

## All available models versus a selected pool

There are two different lists to keep in mind:

- **Available models:** everything you could choose from. This includes the
  existing rule-based registry and the neural players you have exported.
- **A selected pool:** the particular models you choose as possible opponents
  for one training run, evaluation or game session.

For example, you might have 30 rule-based players and 10 exported neural players
available. One experiment could use all 40; another could select only five
rule-based players and three neural players. Having a file in the available
library would not by itself put that player into either experiment.

`players/catalogue.py` is one shared Python module. You do not write another
`catalogue.py` for every pool. Separate JSON files list the selected model
identities. The shared module finds those models and creates the players each
program needs.

There are also two levels of selection: first choose which models belong to the
pool; then the game runner samples that game's opponents from that pool.

For example, `configs/training_pool.json` and `configs/evaluation_pool.json`
can select different lists from the same available library. Creating these files
does not copy the weights or create new models.

## Choose a pool after exporting

First, run the [export command](#export-command) for each checkpoint you want to
use. Then list the available rule-based players and exported neural players:

```powershell
python -m players.catalogue --list-models
```

The list shows model IDs and readable names. For a neural player, use its full
`neural_...` ID in the pool file, not its readable name such as `Neural1`.
Two players can have similar readable names while their IDs identify different
weights.

Create `configs/mixed_pool.json` with the following content. Replace
`neural_<full hash>` with the complete ID printed by the exporter or listing:

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

This pool contains four possible opponents. It does not force all four into
each game. The program using the pool decides how to sample participants.
Keep each ID only once. Empty pools, duplicate IDs and unknown IDs are rejected.
Order matters for reproducing seeded games, so keep the same order when you
want to repeat an experiment.

By default, neural IDs refer to folders under the project's
`players/neural/weights/`. If you exported to a different library, add
`weights_root`:

```json
{
  "weights_root": "../saved_neural_players",
  "model_ids": [
    "RandomPlayer",
    "ThresholdPlayer1",
    "RankingPlayer7",
    "neural_<full hash>"
  ]
}
```

A relative `weights_root` starts from the folder containing the pool JSON.
For `configs/mixed_pool.json`, `../saved_neural_players` therefore means
`D:\Startups\saved_neural_players`. You can also use an absolute path. List the
contents of that library with:

```powershell
python -m players.catalogue --list-models --weights-root saved_neural_players
```

The command-line `--weights-root` above is relative to your current folder;
the `weights_root` inside a pool file is relative to that file.

Only the selected neural models are loaded for play. Their weights stay fixed,
and each occupied seat gets a separate player controller. Loading the pool once
lets later games reuse its loaded weights.

## Play against the selected pool

After saving the pool file, start the existing human interface with:

```powershell
python run_game.py --catalog-file configs/mixed_pool.json --seed 42
```

The program shows the selected pool, then independently samples three opponents
from it. Each entry has the same chance on each draw, so the same model can
occupy more than one seat. Your seat and the starting player are randomized.
The interface still shows your observation and legal choices, waits for your
input, and pauses after each opponent's complete turn. Choosing another game
draws new opponents from the same selected pool.

Without `--catalog-file`, the command keeps using the existing rule-based pool:

```powershell
python run_game.py --seed 42
```

Neural opponents run on CPU by default. To request a different device, add
`--opponent-device cuda` or `--opponent-device auto`:

```powershell
python run_game.py --catalog-file configs/mixed_pool.json --opponent-device cuda
```

The pool does not change during the session when you export another model or
edit the JSON file. Start a new session with an updated file to use a new pool.
Training can [automatically admit qualifying neural opponents](training.md#train-with-a-growing-opponent-pool).
That feature does not alter an already-running human-play session.

To compare automated players without taking a human seat, follow
[evaluation with a mixed pool](evaluation.md#evaluate-a-selected-mixed-pool).
To train a model against them, follow
[training with a selected opponent pool](training.md#choose-the-opponents-for-training).

## Which script does what?

| Script | Responsibility |
| --- | --- |
| [artifacts.py](../players/neural/artifacts.py) | Save permanent player entries, check and load them, and create player controllers. Other code can call these functions directly. |
| [export_model.py](../players/neural/export_model.py) | Provide the command that reads an existing checkpoint and uses the saving functions to export its online network. |
| [catalogue.py](../players/catalogue.py) | List available models and turn a selected pool file into players, loading each selected neural model once. |
| [checkpoints.py](../training/checkpoints.py) | Read the supported training checkpoint formats. Existing training recovery remains separate from extracting a network for export. |

For training recovery and continuation, see
[training, evaluation and recovery](training.md). The permanent player files are
for playing games; `latest.pt` remains the file used to recover training.
