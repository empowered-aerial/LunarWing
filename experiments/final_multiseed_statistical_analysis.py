
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INPUT = RESULTS / "generalization_multiseed_evaluation.csv"
EXPECTED_MODELS = {
    f"{condition}_seed{seed}"
    for condition in ("fixed", "randomized")
    for seed in (7, 123, 1001, 2026)
}
EXPECTED_SCENARIOS = 10
EXPECTED_EVALUATION_SEEDS = set(range(100, 110))
T_CRITICAL_DF3 = 3.182446305

if not INPUT.exists():
    raise FileNotFoundError(f"Missing evaluation results: {INPUT}")

df = pd.read_csv(INPUT)

required = {
    "model", "scenario", "reward", "episode_length",
    "landing_success", "reward_ge_200"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

seed_column = next(
    (column for column in ("seed", "evaluation_seed") if column in df.columns),
    None
)
if seed_column is None:
    raise ValueError("Could not find evaluation seed column: expected seed or evaluation_seed")

df["model"] = df["model"].astype(str)
df["scenario"] = df["scenario"].astype(str)
df["training_seed"] = df["model"].str.extract(r"_seed(\d+)$")[0]
if df["training_seed"].isna().any():
    raise ValueError("Could not extract training seed from one or more model names")

df["training_seed"] = df["training_seed"].astype(int)
df["condition"] = df["model"].str.extract(r"^(fixed|randomized)_")[0]
df[seed_column] = pd.to_numeric(df[seed_column], errors="raise")
df["reward"] = pd.to_numeric(df["reward"], errors="raise")
df["episode_length"] = pd.to_numeric(df["episode_length"], errors="raise")

def as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        raise ValueError("Boolean outcome column contains missing values")
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized Boolean value: {value!r}")

df["landing_success"] = df["landing_success"].map(as_bool)
df["reward_ge_200"] = df["reward_ge_200"].map(as_bool)

actual_models = set(df["model"].unique())
if actual_models != EXPECTED_MODELS:
    raise ValueError(
        f"Model mismatch. Missing: {sorted(EXPECTED_MODELS - actual_models)}; "
        f"unexpected: {sorted(actual_models - EXPECTED_MODELS)}"
    )

if df.duplicated(["model", "scenario", seed_column]).any():
    raise ValueError("Duplicate model/scenario/evaluation-seed records found")

if df["scenario"].nunique() != EXPECTED_SCENARIOS:
    raise ValueError(f"Expected {EXPECTED_SCENARIOS} scenarios, found {df['scenario'].nunique()}")

if set(df[seed_column].astype(int).unique()) != EXPECTED_EVALUATION_SEEDS:
    raise ValueError("Evaluation seeds do not match the expected range 100–109")

episode_counts = df.groupby("model").size()
if not (episode_counts == 100).all():
    raise ValueError(f"Expected 100 episodes per model; got {episode_counts.to_dict()}")

scenario_counts = df.groupby("model")["scenario"].nunique()
if not (scenario_counts == EXPECTED_SCENARIOS).all():
    raise ValueError("One or more models do not cover all expected scenarios")

model_summary = (
    df.groupby(["condition", "training_seed", "model"], as_index=False)
    .agg(
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        mean_episode_length=("episode_length", "mean"),
        reward_ge_200_count=("reward_ge_200", "sum"),
        landing_successes=("landing_success", "sum"),
        evaluation_episodes=("reward", "size"),
    )
    .sort_values(["condition", "training_seed"])
)

condition_rows = []
for condition, group in model_summary.groupby("condition"):
    values = group["mean_reward"].to_numpy(dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    margin = T_CRITICAL_DF3 * sd / math.sqrt(n) if n == 4 else float("nan")
    condition_rows.append({
        "condition": condition,
        "training_runs": n,
        "mean_of_run_mean_rewards": mean,
        "sd_across_training_runs": sd,
        "ci95_low_t": mean - margin,
        "ci95_high_t": mean + margin,
        "mean_landing_successes_per_100_episodes": float(
            group["landing_successes"].mean()
        ),
        "total_landing_successes": int(group["landing_successes"].sum()),
        "total_evaluation_episodes": int(group["evaluation_episodes"].sum()),
    })

condition_summary = pd.DataFrame(condition_rows)

wide = model_summary.pivot(
    index="training_seed", columns="condition", values="mean_reward"
)
if not {"fixed", "randomized"}.issubset(wide.columns) or wide[["fixed", "randomized"]].isna().any().any():
    raise ValueError("Could not construct complete matched training-seed comparisons")

wide["randomized_minus_fixed"] = wide["randomized"] - wide["fixed"]
differences = wide["randomized_minus_fixed"].to_numpy(dtype=float)
n_pairs = len(differences)
mean_difference = float(np.mean(differences))
sd_difference = float(np.std(differences, ddof=1))
difference_margin = T_CRITICAL_DF3 * sd_difference / math.sqrt(n_pairs) if n_pairs == 4 else float("nan")

paired_summary = wide.reset_index()
paired_summary["mean_difference"] = mean_difference
paired_summary["sd_paired_difference"] = sd_difference
paired_summary["ci95_low_t"] = mean_difference - difference_margin
paired_summary["ci95_high_t"] = mean_difference + difference_margin

scenario_summary = (
    df.groupby(["condition", "scenario"], as_index=False)
    .agg(
        mean_reward=("reward", "mean"),
        sd_episode_reward=("reward", "std"),
        episodes=("reward", "size"),
        landing_successes=("landing_success", "sum"),
        reward_ge_200_count=("reward_ge_200", "sum"),
    )
    .sort_values(["scenario", "condition"])
)

model_summary.to_csv(RESULTS / "phase02_model_summary.csv", index=False)
condition_summary.to_csv(RESULTS / "phase02_condition_summary.csv", index=False)
paired_summary.to_csv(RESULTS / "phase02_paired_seed_comparison.csv", index=False)
scenario_summary.to_csv(RESULTS / "phase02_scenario_summary.csv", index=False)

ci_excludes_zero = (
    mean_difference - difference_margin > 0
    or mean_difference + difference_margin < 0
)
conclusion = (
    "The 95% paired t interval excludes zero; interpret cautiously because only four "
    "matched training-seed pairs were evaluated."
    if ci_excludes_zero
    else
    "The 95% paired t interval includes zero; these four matched training-seed pairs "
    "do not establish a reliable difference between training conditions."
)

record = {
    "analysis": "LunarWing Phase 02 multiseed statistical analysis",
    "input": INPUT.name,
    "training_seeds": [7, 123, 1001, 2026],
    "conditions": ["fixed", "randomized"],
    "scenarios": int(df["scenario"].nunique()),
    "evaluation_seeds": sorted(int(x) for x in df[seed_column].unique()),
    "models": int(df["model"].nunique()),
    "total_episodes": int(len(df)),
    "episodes_per_model": 100,
    "primary_metric": "Mean episode reward, averaged within each trained model",
    "comparison": "Paired randomized-minus-fixed differences across matched training seeds",
    "paired_mean_reward_difference": mean_difference,
    "paired_sd_across_training_seed_differences": sd_difference,
    "paired_difference_ci95_t": [
        mean_difference - difference_margin,
        mean_difference + difference_margin,
    ],
    "paired_training_seed_differences": {
        str(int(seed)): float(value)
        for seed, value in zip(wide.index, differences)
    },
    "conclusion": conclusion,
    "limitations": [
        "Only four independent training runs per condition are available.",
        "The t interval relies on assumptions that are difficult to assess with four paired runs.",
        "Episodes within a trained model are not independent training replicates.",
        "Results apply to the tested scenario set and evaluation seeds, not all possible physics.",
        "Reward >= 200 is a proxy and is not interchangeable with the explicit landing_success measure.",
        "Performance variability across training seeds is substantial.",
    ],
}

with open(RESULTS / "phase02_statistical_summary.json", "w", encoding="utf-8") as file:
    json.dump(record, file, indent=2)

report = f"""# LunarWing Phase 02 — Statistical Evidence Record

## Evaluation design

Eight PPO models were evaluated: four fixed-physics and four randomized-physics training runs, using training seeds 7, 123, 1001, and 2026. Each model was evaluated over {EXPECTED_SCENARIOS} physics scenarios and evaluation seeds 100–109, giving {len(df)} episodes in total.

## Primary comparison

The primary metric is each trained model's mean episode reward. The condition-level comparison pairs fixed and randomized runs by training-seed identifier, then computes randomized minus fixed mean reward for each pair.

Mean paired difference: {mean_difference:.3f} reward points.

Paired 95% t interval: [{mean_difference - difference_margin:.3f}, {mean_difference + difference_margin:.3f}].

{conclusion}

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
"""

(RESULTS / "PHASE_02_STATISTICAL_RECORD.md").write_text(report, encoding="utf-8")

print("Validation passed.")
print(f"Models: {df['model'].nunique()} | Episodes: {len(df)}")
print("\nModel summary:")
print(model_summary.round(3).to_string(index=False))
print("\nCondition summary:")
print(condition_summary.round(3).to_string(index=False))
print("\nPaired seed comparison:")
print(paired_summary.round(3).to_string(index=False))
print(f"\nConclusion: {conclusion}")
print("\nGenerated Phase 02 statistical tables and record in:", RESULTS)
