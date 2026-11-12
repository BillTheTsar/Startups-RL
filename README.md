# Startups RL

An experimental reinforcement-learning project built around a playable,
four-player simulator of **Startups**, the card game by Oink Games. It includes
a terminal interface, rule-based opponents, saved neural players, and DQN / Double
DQN training with replay buffers, evaluation and checkpoint recovery.

**Work in progress.** The code and training experiments are still evolving.
This is an unofficial adaptation with custom rules: players start with 15 coins,
minority holders pay every tied majority holder, and coin balances are numeric
and may become negative. See [the simulator](simulator.py) for the implemented rules.

## Quick start

Run commands from the project root. With Conda installed:

```powershell
conda env create -f environment.yml
conda activate Startups
python -m run_game --seed 42
```

This starts a human game against three randomly sampled rule-based opponents.
Enter a numbered action and press Enter to submit it.

[environment.yml](environment.yml) records the Windows/NVIDIA development setup:
Python 3.13, NumPy and PyTorch with CUDA 13.0 support. Neural inference and training
can also use the CPU. The simulator and rule-based play use only the Python
standard library, so they can run without the Conda environment or PyTorch.

### Play or evaluate saved neural players

Exported neural players are included under `players/neural/weights/`. A pool
JSON selects which models participate; it is separate from a training config.
For example, this pool selects 30 rule-based players and 21 neural players:

```powershell
python -m run_game --catalog-file configs/training_pool_1_21.json --seed 42
python -m evaluate --catalog-file configs/training_pool_1_21.json --games 1000 --seed 42 --max-actions 1000
```

Neural opponents use CPU by default; add `--opponent-device cuda` to use an NVIDIA
GPU. Standalone evaluation samples four distinct models per game and writes
results under `results/evaluations/`. An unfinished game at the action limit
scores -1 for each player, then evaluation continues.

### Start training

On a fresh checkout, first build an initial archive of at least 20,000 transitions
(records of player decisions and their outcomes):

```powershell
python -m training.build_buffer --seed 42
python -m training.train --seed 42
```

The buffer builder creates `buffer_data/ranking_player_15.sqlite3` and refuses to
overwrite an existing file. The training command starts the default 10,000-game
DQN run against rule-based opponents, using CUDA when available. Add `--device cpu`
to train on CPU. Outputs go into a new directory under `results/training/`.

Use the [training guide](docs/training.md) to choose a config, enable Double DQN,
LayerNorm or target clipping, select opponents, resume a run, or continue learned
weights in a new run. The files in `configs/` preserve experiment settings;
they are not all intended as recommended starting settings.

### Run the tests

The test suite is kept locally and excluded from Git. If you have the local
`tests/` directory, run it with the Conda environment active:

```powershell
python -m unittest discover -s tests
```

## Read more

Start with the question-based [documentation index](docs/README.md), or go directly to:

- [Training, configuration and recovery](docs/training.md)
- [Evaluating automated players](docs/evaluation.md)
- [Building the initial replay archive](docs/buffer_building.md)
- [Neural inputs and observation encoding](docs/neural_observation_encoder.md)
- [Exporting and selecting neural players](docs/neural_player_exports.md)
- [Recorded experimental observations](docs/experimental_observations.md)

Source code, configs and exported neural players are included in Git.
The test suite (`tests/`) is kept locally and excluded.
Generated replay archives (`buffer_data/`), run outputs (`results/`) and diagnostic
write-ups are excluded. Some historical documentation links refer to those local
artifacts and will not resolve in a fresh checkout.
