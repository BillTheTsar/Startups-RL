# Documentation index

Choose a question below. To start training, begin with
[Training, evaluation and recovery](training.md). Run commands from the project
folder (`D:\Startups`) unless a guide says otherwise.

## Running and managing training

| Your question | Read |
| --- | --- |
| Is training implemented, and how do I start it? | [Start a training run](training.md) |
| How do I start architecture version 2 with LayerNorm from a configuration? | [Train with LayerNorm](training.md#train-with-layernorm-architecture-version-2) |
| How can I track Q-value percentiles and hidden-layer activity during training? | [Optional Q-value monitoring](training.md#track-q-values-during-training) |
| Can I resume an older checkpoint with LayerNorm, and can old and new models share a pool? | [Architecture compatibility](training.md#train-with-layernorm-architecture-version-2) and [saved neural players](neural_player_exports.md#what-is-saved-in-each-entry) |
| How do I start a Double DQN experiment without editing my existing configuration? | [Start fresh with Double DQN](training.md#start-fresh-with-double-dqn) |
| Can I switch existing weights to Double DQN? What happens when I resume? | [Continue with Double DQN](training.md#continue-existing-weights-with-double-dqn) and [resume its saved algorithm](training.md#resume-a-double-dqn-run) |
| How do I change the learning rate, chance of random actions (epsilon), run length or stored training examples? | [Configuration for future runs](training.md#configuration-for-future-runs) and the defaults in [TrainingConfig](../training/config.py) |
| How do I train against a chosen mixture of rule-based and neural opponents? Is the pool file different from the training configuration? | [Choose the opponents for training](training.md#choose-the-opponents-for-training) |
| How do I let successful neural models join the opponent pool while training? | [Growing-pool walkthrough](training.md#train-with-a-growing-opponent-pool) |
| Does `best.pt` mean the highest score or the latest qualifying model? How do admission and demonstration promotion differ? | [Admission and best-model rules](training.md#admission-and-best-model-rules) |
| How do periodic buffer refreshes work, and what happens to RAM-only examples? | [Refresh both buffers from the archive](training.md#refresh-both-buffers-from-the-archive) |
| Does increasing CPU thread counts make training faster? How do I change the setting? | [CPU threads and faster batches](training.md#cpu-threads-and-faster-batches) |
| Which archive records are checked when loading or refreshing replay? What happens to invalid selections? | [Archive validation](training.md#archive-validation) |
| How do I run a short segment or resume an interrupted run? | [Short segments and resume](training.md#short-segments-and-resume) |
| How do I continue finished training with new settings while keeping the learned weights and replay examples? | [Continue finished training with new settings](training.md#continue-finished-training-with-new-settings) |
| Can a new training stage use a different pool? What happens to the pool on exact resume? | [Change opponents in a new stage](training.md#keep-or-change-the-opponent-pool-in-the-new-stage) and [exact resume](training.md#short-segments-and-resume) |
| How often are updates, evaluations and saves performed? | [Training schedule](training.md#training-schedule) and [evaluation and promotion](training.md#evaluation-and-promotion) |
| Where are training outputs written, and what does the directory contain? | [Output directory and layout](training.md#output-directory-and-layout) |
| What are `latest.pt`, `best.pt` and `metrics.json`? Why does the recovery file get larger? | [Run files](training.md#run-files) |
| What happens at the action cap? Does the `-1` penalty count in training and evaluation? | [Training schedule](training.md#training-schedule) and [evaluation and promotion](training.md#evaluation-and-promotion) |
| What happens if the program crashes while saving examples to the database? | [Safe archive updates and recovery](training.md#safe-archive-updates-and-recovery) |
| Which conda environment and GPU packages should I use? | [Environment](neural_training_components.md#environment) and [environment.yml](../environment.yml) |

## Understanding the neural player and learning

| Your question | Read |
| --- | --- |
| What information reaches the model? What do the 332 inputs mean? | [Observation encoder](neural_observation_encoder.md), especially the [exact vector layout](neural_observation_encoder.md#exact-vector-layout) |
| How are past observations, other players' turns and the end of the game represented? | [Historical and terminal views](neural_observation_encoder.md#historical-and-terminal-views) |
| What does the Q-network output, and how does the player select a legal action? | [Network and controller](neural_training_components.md#network-and-controller) |
| How do I save a permanent neural player from `best.pt` or `latest.pt`? | [Export command](neural_player_exports.md#export-command) |
| What is inside each entry under `players/neural/weights/`, and what happens if I export the same model twice? | [Saved files](neural_player_exports.md#what-is-saved-in-each-entry) and [permanent identities](neural_player_exports.md#permanent-identities-and-repeated-exports) |
| Do all saved models become opponents? Do I need a different Python catalogue for each pool? | [Available models versus a selected pool](neural_player_exports.md#all-available-models-versus-a-selected-pool) |
| How do I list all available models, write a pool file, and play against it myself? | [Choose a pool after exporting](neural_player_exports.md#choose-a-pool-after-exporting) and [human play](neural_player_exports.md#play-against-the-selected-pool) |
| What is one training example (a transition)? What do `s`, `a`, reward, next state and the masks mean? | [What one transition means](buffer_building.md#what-one-transition-means) |
| What is the callback, and when does learning happen during a game? | [Completed-transition callbacks](dqn_updates.md#completed-transition-callbacks) |
| How are masked targets, terminal rewards, the loss and target updates calculated? | [DQN update engine](dqn_updates.md#dqn-update-engine) |
| How do I clip training targets, and does that also clip the network's predictions? | [Target clipping and its limits](dqn_updates.md#optional-target-clipping) and [configuration and commands](training.md#clip-training-targets-to-the-possible-return-range) |
| What does Double DQN change? Which network selects the next action and which supplies its value? | [Double DQN example](dqn_updates.md#double-dqn-in-a-small-example) |
| When did the large Q-values and inactive hidden neurons appear in the mixed-opponent run? | [Checkpoint diagnosis](diagnose/checkpoint_timeline.md) |
| Did large prediction errors or weight updates cause the deterioration? What did the bounded diagnostic run find? | [Update diagnosis](diagnose/update_trace.md) and [diagnostic methods and files](diagnose/README.md) |
| How do I connect the pieces for one game in Python? | [One-game usage](dqn_updates.md#one-game-usage) |
| How do `demo` and `non_demo` differ? How are examples chosen for training and removed when buffers fill up? | [Configuration and replay](neural_training_components.md#configuration-and-replay) |
| How was the initial database of examples built, and what does it contain? | [Building the initial training dataset](buffer_building.md), including [rewards and partitions](buffer_building.md#rewards-and-partitions) |
| How are the seven implementation steps divided among modules? | [Modules and tests](training.md#modules-and-tests) |

## Rule-based opponents and experiments

| Your question | Read |
| --- | --- |
| How do threshold and threshold-chicken players decide? | [Parameterized threshold players](threshold_players.md) |
| How do ranking players score actions, and how can I tune their weights? | [WeightedRankingPlayer: scoring and tuning](weighted_ranking_player.md) |
| How can I inspect why a ranking player chose an action? | [Inspecting scores](weighted_ranking_player.md#inspecting-scores-without-changing-the-choice-sequence) |
| How do I compare automated players outside a training run? | [Evaluating automated players](evaluation.md) |
| How do I evaluate a selected mixture of exported neural and rule-based players? | [Evaluate a selected mixed pool](evaluation.md#evaluate-a-selected-mixed-pool) |
| How are placements, points and the comparison matrix calculated? | [Points, wins and placements](evaluation.md#points-wins-and-placements) and the [head-to-head matrix](evaluation.md#normalized-head-to-head-matrix) |
| What did earlier experiments suggest? | [Experimental observations](experimental_observations.md); these are dated results and interpretations, not the current training settings |

`evaluation.md` describes the separate tool for comparing players. To understand
the evaluations run during DQN training, read
[training.md](training.md#evaluation-and-promotion).

## Game implementation

These topics are explained in the documentation inside the source files
(docstrings):

| Your question | Read |
| --- | --- |
| Where are setup, legal moves, end-of-game scoring and the full game state implemented? | [simulator.py](../simulator.py) |
| What can each player see, and what information is known to everyone? | [simulator.py](../simulator.py) and the [encoder completeness check](neural_observation_encoder.md#completeness-audit-and-deliberate-omissions) |
| How are the 41 action IDs and legal masks represented? | [action_codec.py](../action_codec.py) |
| What does the game loop do, and what does the human command-line interface do? | [game_loop.py](../game_loop.py) and [run_game.py](../run_game.py) |
| What must a `Player` implement, and what does `on_observation` do? | [players/base.py](../players/base.py) |

To understand the code, read the encoder guide, neural components, DQN updates,
then the training guide. To run an experiment, start with the training guide:
it explains starting, changing settings, inspecting results and resuming.
