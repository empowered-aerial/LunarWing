# Project LunarWing — Experimental Record

## Research question

Does training PPO across randomized LunarLander physics improve performance on previously unseen physics conditions compared with training under fixed nominal physics?

## Experimental design

- Environment: Gymnasium LunarLander, Gymnasium 1.4.0.
- RL algorithm: Proximal Policy Optimization (PPO), using Stable-Baselines3.
- Training conditions: fixed nominal physics versus randomized physics.
- Independent training seeds per condition: 42, 123, 2026.
- Training budget: approximately 100,000 timesteps per model (100,352 actual timesteps).
- Evaluation: 10 physics scenarios, with evaluation seeds 100–109.
- Evaluation policy: deterministic actions.
- Models: six total, three per training condition.
- Evaluation dataset: 600 episodes, 100 per model.

## Training configuration

Shared PPO parameters:
- Learning rate: 0.0003
- n_steps: 1024
- Batch size: 64
- Gamma: 0.99
- GAE lambda: 0.95
- Entropy coefficient: 0
- Device: CPU

Fixed-physics training:
- Gravity: -10
- Wind disabled

Randomized-physics training:
- Gravity sampled from [-11, -9]
- Wind power sampled from [0, 15]
- Turbulence power sampled from [0, 1.5]
- Wind enabled

## Data and analysis files

Corrected evaluation data:
- results/generalization_multiseed_evaluation.csv
- results/generalization_paired_reward_differences.csv

Validated summaries:
- results/generalization_multiseed_model_summary_validated.csv
- results/generalization_multiseed_by_scenario_validated.csv
- results/generalization_multiseed_summary_validated.json

Final analysis tables:
- results/final_tables/model_summary.csv
- results/final_tables/scenario_summary.csv
- results/final_tables/paired_reward_differences.csv
- results/final_tables/paired_reward_confidence_intervals.csv
- results/final_tables/scenario_reward_differences.csv

Final figures:
- results/final_figures/model_mean_rewards.png
- results/final_figures/paired_reward_differences.png
- results/final_figures/scenario_performance.png

## Data integrity and termination-label correction

The corrected evaluation dataset contains 600 rows, six unique models, 10 scenarios, and 10 evaluation seeds. Each model has 100 episodes and each model/scenario combination has 10 episodes. No duplicate model/scenario/seed combinations or missing values were found in the required metrics.

A termination-classification discrepancy was identified during validation. Gymnasium can terminate an episode when the lander moves out of bounds even when the internal game_over flag is false. The evaluator was corrected to classify out-of-bounds episodes using the saved final horizontal position, while retaining the existing landing and time-limit classifications.

The correction changed termination labels, not the underlying recorded rewards or trajectory metrics. Relevant invariant metrics were compared with pre-correction backups, and the integrity audit passed.

## Main findings

Across 300 paired evaluation episodes per training condition:
- Fixed-physics models: 8 verified successful landings (2.67%).
- Randomized-physics models: 7 verified successful landings (2.33%).

Mean paired reward differences (randomized minus fixed):
- Training seed 42: -16.449; 95% cluster-bootstrap CI [-48.106, 16.133].
- Training seed 123: +49.627; 95% cluster-bootstrap CI [35.943, 62.255].
- Training seed 2026: -73.552; 95% cluster-bootstrap CI [-106.580, -40.374].

The mean of the three training-seed reward differences is -13.458, with a sample standard deviation of 61.644.

Scenario-level mean reward differences across the three training seeds ranged from -32.619 to +24.322. The direction and magnitude of the observed differences varied by scenario and training seed.

## Interpretation

The experiment does not demonstrate a consistent generalization advantage from randomized-physics training under the tested configuration and training budget.

Results show substantial sensitivity to training seed. Successful landings were rare for both conditions. The findings should be treated as preliminary evidence about these specific configurations, not as proof that physics randomization is ineffective in general.

## Statistical limitations

- Only three independent training seeds were evaluated per condition.
- The cluster-bootstrap intervals resample evaluation seed IDs within each training seed, retaining the associated scenarios together. They describe evaluation-level uncertainty conditional on each trained model; they do not quantify uncertainty over the full population of possible training runs.
- Wilson intervals calculated from pooled episodes do not account for variation between independently trained models.
- Scenario comparisons average only three training-seed-specific differences and are descriptive.
- Reward is not interchangeable with verified landing success.
- Evaluation covers 10 selected scenarios and does not establish robustness to every possible physical condition.

## Reproducibility notes

- Preserve original and corrected CSV backups.
- Do not overwrite existing backup files.
- Record any changes to environment versions, model configurations, evaluation code, or termination classification.
- Re-run the data-integrity audit after any changes to evaluation data.
- Interpret results using the corrected validated files, not earlier summaries produced before the termination-label correction.

## Scientific conclusion

This experiment provides a reproducible comparison framework and identifies training-seed sensitivity as a major issue. Further independent training runs, stronger landing performance, and broader evaluation would be needed to establish whether randomized physics improves generalization reliably.

