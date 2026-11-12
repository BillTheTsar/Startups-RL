# Experimental observations

This records evaluation findings and our working explanations in date order.
Each entry uses its run ID as the heading and refers to that run's saved
configurations and opponent pool (the set of models available to play).
Later entries may confirm, refine, or contradict earlier ones. Keep earlier
entries as a record of what we understood at the time, and link back to them
when discussing new evidence.

## 20261030T135800Z_e6c90c91a9d6

**Experiment:** 200,000 completed four-player games with 30 model configurations.
Each game used four different models, chosen with equal sampling chances, and
gave each player an equal chance to start.

**Sources:** [Report](../results/evaluations/20261030T135800Z_e6c90c91a9d6/report.txt),
[unrounded statistics](../results/evaluations/20261030T135800Z_e6c90c91a9d6/summary.json),
and [configuration manifest](../results/evaluations/20261030T135800Z_e6c90c91a9d6/manifest.json).

An **outright win** means finishing alone with the most coins. **Evaluation
points** measure finishing position and are separate from coins; see
[Points, wins, and placements](evaluation.md#points-wins-and-placements).

### Observations

1. **Starting first or fourth shows a slight advantage; starting second shows
   a slight disadvantage.** Combining the results for all models gives the
   following results. Each position has exactly 200,000 appearances, and positions are
   turn-order positions counted from the starter, rather than fixed seat IDs.

   | Starting position | Outright win rate | Total evaluation points | Mean points per game |
   |---|---:|---:|---:|
   | First | 24.2685% | +1,370.5 | +0.00685 |
   | Second | 23.5295% | -3,399.0 | -0.01700 |
   | Third | 23.7505% | -52.0 | -0.00026 |
   | Fourth | 24.1695% | +2,080.5 | +0.01040 |

   First has the highest outright win rate; fourth has the highest mean points.
   The differences between first and fourth are too small to establish a clear
   ordering in this sample. Second is the weakest position overall and has
   the lowest observed mean points for 18 of the 30 models. Giving each model
   equal weight keeps the overall pattern. The effect is small and varies
   by strategy; it is not a universal ordering for every model.

2. **M26 leads overall and finishes above each rival more often than below it
   in their shared games.** M26 is `RankingPlayer15` in this run. Across 26,485
   appearances, it earned **+14,262 total points**, the highest **mean points of +0.53849**,
   and an **outright win rate of 41.8954%**.

   All **29** of its matchup scores against other models are positive.
   Its narrowest advantage is against M23 (`RankingPlayer12`), at **+0.06367**;
   its largest is against M1 (`RandomPlayer`), at **+0.72840**. These values are
   `(pairwise wins - pairwise losses) / shared games`: a pairwise win means
   finishing above that rival, and a pairwise loss means finishing below it.
   Tied finishes count in the number of shared games. Thus, M26 finishes above
   each rival more often than it finishes below that rival in their shared games.

   This was surprising: we expected even the strongest model overall to have
   a "kryptonite" rival against which it performed worse on average. No such
   rival appears among the other 29 models in this experiment.

### Interpretation and open questions

The starting-position pattern and M26's positive matchup results apply to this
sampled opponent pool and these saved configurations. The matchup table compares
two models' finishes within four-player games, so the other two opponents can
affect each comparison. Positive scores in this run do not show that a strategy
beats every possible opponent or will get the same results against new opponents.

Future entries can investigate whether the position effect continues with new
seeds or different opponent pools, and whether M26 retains positive matchups
against stronger strategies or ones deliberately chosen for a different style.
If a conclusion changes, identify the new evidence and link back to this entry.

**Relation to earlier entries:** First entry; no earlier observations to update.
