
import json
import shutil
from pathlib import Path

import pandas as pd

results_dir = Path("results")
episode_path = results_dir / "generalization_evaluation.csv"
summary_path = results_dir / "generalization_summary.json"
scenario_path = results_dir / "generalization_by_scenario.csv"

paths = [episode_path, summary_path, scenario_path]

for path in paths:
    backup = path.with_name(
        f"{path.stem}_before_termination_fix{path.suffix}"
    )
    if not backup.exists():
        shutil.copy2(path, backup)

results = pd.read_csv(episode_path)
metadata = json.loads(summary_path.read_text(encoding="utf-8"))

original_other = results["termination_reason"].eq("other_termination")
terminated = (
    results["terminated"].astype(str).str.lower().eq("true")
)
out_of_bounds = (
    pd.to_numeric(results["final_horizontal_position"], errors="coerce")
    .abs()
    >= 1.0
)

corrected = original_other & terminated & out_of_bounds
results.loc[corrected, "termination_reason"] = "crash_or_out_of_bounds"

scenario_summary = (
    results.groupby(["model", "scenario"])
    .agg(
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        mean_episode_length=("episode_length", "mean"),
        episodes_reward_ge_200=("reward_ge_200", "sum"),
        landing_successes=("landing_success", "sum"),
        crashes=(
            "termination_reason",
            lambda s: (s == "crash_or_out_of_bounds").sum(),
        ),
        time_limits=(
            "termination_reason",
            lambda s: (s == "time_limit").sum(),
        ),
        other_terminations=(
            "termination_reason",
            lambda s: (s == "other_termination").sum(),
        ),
        evaluation_episodes=("reward", "size"),
    )
    .reset_index()
)

summary = (
    results.groupby("model")
    .agg(
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        mean_episode_length=("episode_length", "mean"),
        episodes_reward_ge_200=("reward_ge_200", "sum"),
        landing_successes=("landing_success", "sum"),
        crashes=(
            "termination_reason",
            lambda s: (s == "crash_or_out_of_bounds").sum(),
        ),
        time_limits=(
            "termination_reason",
            lambda s: (s == "time_limit").sum(),
        ),
        other_terminations=(
            "termination_reason",
            lambda s: (s == "other_termination").sum(),
        ),
        evaluation_episodes=("reward", "size"),
    )
    .reset_index()
)

results.to_csv(episode_path, index=False)
scenario_summary.to_csv(scenario_path, index=False)

metadata["model_summary"] = summary.to_dict(orient="records")
metadata["scenario_summary"] = scenario_summary.to_dict(orient="records")
summary_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print(f"Corrected out-of-bounds classifications: {int(corrected.sum())}")
print(
    "Remaining other terminations:",
    int(results["termination_reason"].eq("other_termination").sum()),
)
print("\nCorrected overall summary:")
print(summary.to_string(index=False))
print("\nUpdated files:")
for path in paths:
    print(path)
print("\nOriginal files backed up with '_before_termination_fix' suffix.")
