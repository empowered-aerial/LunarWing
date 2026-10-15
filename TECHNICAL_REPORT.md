# LunarWing Technical Report

**Title:** Robust Autonomous Lunar Landing Under Uncertain Physics  
**Author:** Mohammed Jasim Salim  
**Artifact status:** Research implementation and recorded experiments packaged for reproducibility.

## Abstract

LunarWing investigates whether exposing reinforcement-learning agents to randomized environment physics during training improves generalization to held-out conditions. The experimental pipeline compares fixed-physics and randomized-physics PPO policies, evaluates saved policies across multiple scenarios and seeds, records episode-level outcomes, and summarizes reward differences and landing-related metrics. A separate custom PyTorch episodic actor–critic was implemented as an additional learning experiment.

## Research questions

1. Does randomized-physics training improve performance on tested unseen physics conditions relative to fixed-physics training?
2. Are the observed differences consistent across training seeds and evaluation scenarios?

## Experimental design

The primary environment is Gymnasium `LunarLander-v3`. The repository contains fixed-physics and randomized-physics PPO checkpoints, multi-seed training records, scenario-level evaluation data, paired reward comparisons, statistical summaries, and failure-taxonomy analysis.

The primary evaluation artifacts are the curated tables under `results/final_tables/`, the figures under `results/final_figures/`, and the corresponding experiment records. The raw episode-level data should be used to independently recompute aggregate statistics.

## Metrics

Recorded metrics include episode reward, episode length, explicit landing-success outcomes, reward-threshold counts, crash and termination categories, and physics-scenario parameters. Reward alone is not treated as proof of successful landing.

## Results and interpretation

Consult `results/final_tables/model_summary.csv` for the validated model-level summary, `results/final_tables/paired_reward_confidence_intervals.csv` for paired reward comparisons, and `results/final_tables/scenario_summary.csv` for scenario-level outcomes. Figures are provided in `results/final_figures/`.

The conclusions must be restricted to the recorded seeds, evaluation scenarios, and experimental settings. The existence of a randomized training condition does not by itself establish superior generalization; direction, uncertainty, and practical significance must be judged from the validated tables. Historical intermediate files are retained for provenance and should not be substituted for corrected final results.

## Phase 04: custom actor–critic

Phase 04 implemented and ran a standalone vanilla episodic actor–critic in PyTorch on LunarLander-v3 for 100 episodes with seed 42, Adam learning rate 0.0003, and discount factor 0.99. The initial deterministic evaluation mean reward was -291.595; the final evaluation mean reward was -176.323. The run is implementation-complete but does not establish a successful learned landing policy. Training logs and curves are in `results/phase04_actor_critic/`.

## Reproducibility

The repository contains experiment scripts, saved model checkpoints, result tables, episode-level CSVs, JSON summaries, plots, and phase records. `requirements.txt` records packages from the existing project virtual environment. Reproduction requires installing compatible dependencies, running the environment tests, inspecting script arguments and seeds, and executing the selected training and evaluation commands.

A successful test run and a clean-environment reproduction should be recorded before claiming full independent reproducibility. Software and hardware differences may affect training results.

## Limitations

- Findings are bounded by the physics scenarios and seeds actually evaluated.
- Reward and reward thresholds are imperfect proxies for safe landing.
- Reinforcement-learning results can vary with seeds, dependencies, and hardware.
- The custom actor–critic experiment did not establish successful landing behavior.
- Historical files are retained, so readers must use validated summaries and experiment records to identify final results.

## Conclusion

LunarWing provides a collection of reinforcement-learning experiments and recorded evidence for studying policy generalization under physics variation. Its public value depends on transparent metric definitions, correct identification of validated results, and successful independent reproduction. Any comparative claim should be supported by the final tables and uncertainty estimates rather than by training completion alone.
