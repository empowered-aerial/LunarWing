# LunarWing Phase 02 — Statistical Evidence Record

## Evaluation design

Eight PPO models were evaluated: four fixed-physics and four randomized-physics training runs, using training seeds 7, 123, 1001, and 2026. Each model was evaluated over 10 physics scenarios and evaluation seeds 100–109, giving 800 episodes in total.

## Primary comparison

The primary metric is each trained model's mean episode reward. The condition-level comparison pairs fixed and randomized runs by training-seed identifier, then computes randomized minus fixed mean reward for each pair.

Mean paired difference: -36.915 reward points.

Paired 95% t interval: [-129.724, 55.895].

The 95% paired t interval includes zero; these four matched training-seed pairs do not establish a reliable difference between training conditions.

## Outcome interpretation

The results do not justify claiming that randomized-physics training consistently improves generalization. Results differ substantially across training seeds. Actual landing successes are reported separately from the reward >= 200 proxy and should not be treated as equivalent outcomes.

## Limitations

- Four training runs per condition provide limited precision and weak support for distributional assumptions.
- The paired t interval is an uncertainty estimate, not proof of causation or universal generalization.
- The evaluation covers the specified scenarios and seeds only.
- Episode-level variation does not replace independent training runs as the unit for comparing training conditions.
- The low frequency of explicit successful landings limits strong claims about landing reliability.
- The conclusions are specific to the implemented environment, PPO configuration, training budget, and evaluation design.

## Validation and provenance

The analysis script checks expected model identities, unique model/scenario/evaluation-seed records, all ten scenarios, evaluation seeds 100–109, and 100 episodes per model before generating outputs. The raw multiseed evaluation CSV is retained as the analysis input.

Generated tables:
- `phase02_model_summary.csv`
- `phase02_condition_summary.csv`
- `phase02_paired_seed_comparison.csv`
- `phase02_scenario_summary.csv`
- `phase02_statistical_summary.json`

This record documents the current evidence; it does not claim that all possible sources of uncertainty have been eliminated.
