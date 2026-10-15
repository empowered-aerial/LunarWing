
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_PATH = Path("results/generalization_evaluation.csv")
BOOTSTRAP_REPS = 20000
RANDOM_SEED = 2026

df = pd.read_csv(RESULTS_PATH)
df = df[df["model"].isin(["fixed_100k", "randomized_100k"])]

paired = df.pivot(
    index=["scenario", "seed"],
    columns="model",
    values="reward",
).reset_index()

paired["difference"] = (
    paired["randomized_100k"] - paired["fixed_100k"]
)

scenario_means = (
    paired.groupby("scenario")["difference"]
    .mean()
    .sort_index()
)

overall_mean = paired["difference"].mean()
scenario_sd = scenario_means.std(ddof=1)
scenario_count = len(scenario_means)
t_critical_95 = 2.262157  # 95% two-sided t critical value, 9 df

scenario_ci_low = (
    overall_mean
    - t_critical_95 * scenario_sd / np.sqrt(scenario_count)
)
scenario_ci_high = (
    overall_mean
    + t_critical_95 * scenario_sd / np.sqrt(scenario_count)
)

rng = np.random.default_rng(RANDOM_SEED)
scenario_values = {
    name: group["difference"].to_numpy()
    for name, group in paired.groupby("scenario")
}
scenario_names = list(scenario_values)
bootstrap_means = np.empty(BOOTSTRAP_REPS)

for i in range(BOOTSTRAP_REPS):
    sampled_names = rng.choice(
        scenario_names,
        size=len(scenario_names),
        replace=True,
    )
    sampled_differences = np.concatenate(
        [scenario_values[name] for name in sampled_names]
    )
    bootstrap_means[i] = sampled_differences.mean()

bootstrap_ci_low, bootstrap_ci_high = np.percentile(
    bootstrap_means, [2.5, 97.5]
)

naive_bootstrap_means = np.empty(BOOTSTRAP_REPS)
differences = paired["difference"].to_numpy()

for i in range(BOOTSTRAP_REPS):
    sample = rng.choice(
        differences,
        size=len(differences),
        replace=True,
    )
    naive_bootstrap_means[i] = sample.mean()

naive_ci_low, naive_ci_high = np.percentile(
    naive_bootstrap_means, [2.5, 97.5]
)

randomized_wins = int((differences > 0).sum())
fixed_wins = int((differences < 0).sum())
ties = int((differences == 0).sum())

print("FINAL PAIRED REWARD ANALYSIS")
print(f"Matched scenario-seed pairs: {len(paired)}")
print(f"Scenarios: {scenario_count}")
print(f"Mean reward difference (randomized - fixed): {overall_mean:.3f}")
print(f"Randomized wins: {randomized_wins}")
print(f"Fixed wins: {fixed_wins}")
print(f"Ties: {ties}")

print("\n95% CONFIDENCE INTERVALS")
print(
    "Scenario-level t interval: "
    f"[{scenario_ci_low:.3f}, {scenario_ci_high:.3f}]"
)
print(
    "Scenario-cluster bootstrap interval: "
    f"[{bootstrap_ci_low:.3f}, {bootstrap_ci_high:.3f}]"
)
print(
    "Naive paired bootstrap interval (assumes pairs independent): "
    f"[{naive_ci_low:.3f}, {naive_ci_high:.3f}]"
)

print("\nMEAN REWARD DIFFERENCE BY SCENARIO")
print(scenario_means.round(3).to_string())

print("\nInterpretation guide:")
if bootstrap_ci_high < 0:
    print(
        "The scenario-cluster bootstrap interval is below zero: "
        "the results favor fixed-physics training across the tested scenarios."
    )
elif bootstrap_ci_low > 0:
    print(
        "The scenario-cluster bootstrap interval is above zero: "
        "the results favor randomized-physics training across the tested scenarios."
    )
else:
    print(
        "The scenario-cluster bootstrap interval includes zero: "
        "the aggregate difference remains uncertain under this analysis."
    )

print(
    "\nThese intervals quantify uncertainty under the selected resampling "
    "assumptions. They do not account for variation across independent "
    "training runs, because only one trained model per condition was evaluated."
)
