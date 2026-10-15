import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results" / "generalization_multiseed_evaluation.csv"
OUT_CSV = ROOT / "results" / "phase03_failure_taxonomy.csv"
OUT_MD = ROOT / "results" / "phase03_failure_taxonomy.md"

EXPECTED_MODELS = 8
EXPECTED_EPISODES_PER_MODEL = 100
REQUIRED = {
    "model", "seed", "scenario", "reward", "episode_length",
    "reward_ge_200", "terminated", "truncated", "termination_reason",
    "landing_success", "left_leg_contact", "right_leg_contact",
    "final_horizontal_position", "final_vertical_position",
    "final_horizontal_velocity", "final_vertical_velocity",
    "final_angle", "final_angular_velocity"
}

def boolean(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}")

def classify(row):
    reason = row["termination_reason"].strip().lower()
    landed = boolean(row["landing_success"])
    terminated = boolean(row["terminated"])
    truncated = boolean(row["truncated"])

    if landed:
        if reason not in {"successful_landing", ""}:
            return "landing_reason_conflict"
        return "successful_landing"

    if reason == "successful_landing":
        return "landing_flag_conflict"
    if reason == "crash_or_out_of_bounds":
        return "crash_or_out_of_bounds"
    if reason == "time_limit":
        return "timeout"
    if reason == "other_termination":
        return "other_termination"
    if not terminated and truncated:
        return "truncated_unclassified"
    if not terminated and not truncated:
        return "episode_end_unclassified"
    return "termination_unclassified"

def condition_of(model):
    name = model.lower()
    if name.startswith("fixed"):
        return "fixed"
    if name.startswith("randomized"):
        return "randomized"
    return "unknown"

def mean(values):
    return sum(values) / len(values) if values else None

def fmt(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"

if not INPUT.exists():
    sys.exit(f"ERROR: Input CSV not found: {INPUT}")

if OUT_CSV.exists() or OUT_MD.exists():
    sys.exit(
        "ERROR: At least one output already exists. "
        "Nothing was overwritten. Inspect existing outputs before proceeding."
    )

with INPUT.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    fields = set(reader.fieldnames or [])
    missing = REQUIRED - fields
    if missing:
        sys.exit(f"ERROR: Missing required columns: {sorted(missing)}")
    rows = list(reader)

if not rows:
    sys.exit("ERROR: Input CSV contains no episodes.")

models = sorted({row["model"] for row in rows})
if len(models) != EXPECTED_MODELS:
    sys.exit(f"ERROR: Expected {EXPECTED_MODELS} models, found {len(models)}.")

counts_per_model = Counter(row["model"] for row in rows)
bad_counts = {
    model: count for model, count in counts_per_model.items()
    if count != EXPECTED_EPISODES_PER_MODEL
}
if bad_counts:
    sys.exit(f"ERROR: Unexpected episode counts per model: {bad_counts}")

unique_keys = [
    (row["model"], row["scenario"], row["seed"])
    for row in rows
]
if len(set(unique_keys)) != len(unique_keys):
    sys.exit("ERROR: Duplicate model/scenario/evaluation-seed records found.")

for row in rows:
    row["_condition"] = condition_of(row["model"])
    row["_category"] = classify(row)
    for key in (
        "reward", "episode_length", "final_horizontal_position",
        "final_vertical_position", "final_horizontal_velocity",
        "final_vertical_velocity", "final_angle", "final_angular_velocity"
    ):
        row[f"_{key}"] = float(row[key])
    for key in ("landing_success", "reward_ge_200",
                "left_leg_contact", "right_leg_contact"):
        row[f"_{key}"] = boolean(row[key])

categories = [
    "successful_landing",
    "crash_or_out_of_bounds",
    "timeout",
    "other_termination",
    "truncated_unclassified",
    "episode_end_unclassified",
    "termination_unclassified",
    "landing_reason_conflict",
    "landing_flag_conflict",
]

groups = [("overall", "all", rows)]
for condition in ("fixed", "randomized"):
    subset = [r for r in rows if r["_condition"] == condition]
    if subset:
        groups.append(("condition", condition, subset))
for model in models:
    groups.append(("model", model, [r for r in rows if r["model"] == model]))

summary = []
for group_type, group_name, subset in groups:
    counts = Counter(r["_category"] for r in subset)
    for category in categories:
        n = counts[category]
        if n == 0:
            continue
        selected = [r for r in subset if r["_category"] == category]
        summary.append({
            "group_type": group_type,
            "group": group_name,
            "category": category,
            "episodes": n,
            "percentage": round(100 * n / len(subset), 2),
            "mean_reward": round(mean([r["_reward"] for r in selected]), 3),
            "mean_episode_length": round(mean([r["_episode_length"] for r in selected]), 3),
            "reward_ge_200_count": sum(r["_reward_ge_200"] for r in selected),
            "left_leg_contact_count": sum(r["_left_leg_contact"] for r in selected),
            "right_leg_contact_count": sum(r["_right_leg_contact"] for r in selected),
            "explicit_landing_success_count": sum(r["_landing_success"] for r in selected),
        })

with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

overall = Counter(r["_category"] for r in rows)
fixed = [r for r in rows if r["_condition"] == "fixed"]
randomized = [r for r in rows if r["_condition"] == "randomized"]

lines = [
    "# Phase 03 — Step 1: Failure Taxonomy",
    "",
    "## Dataset validation",
    "",
    f"- Input: `{INPUT.relative_to(ROOT)}`",
    f"- Total episodes: {len(rows)}",
    f"- Models: {len(models)}",
    f"- Episodes per model: {EXPECTED_EPISODES_PER_MODEL}",
    f"- Unique model/scenario/evaluation-seed combinations: {len(set(unique_keys))}",
    "",
    "## Overall episode outcomes",
    "",
    "| Outcome category | Episodes | Percentage |",
    "|---|---:|---:|",
]
for category in categories:
    n = overall[category]
    if n:
        lines.append(f"| {category} | {n} | {100*n/len(rows):.2f}% |")

lines += [
    "",
    "## Fixed versus randomized training",
    "",
    "| Training condition | Episodes | Successful landings | Crashes/out of bounds | Timeouts | Other/unclassified |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, subset in (("fixed", fixed), ("randomized", randomized)):
    c = Counter(r["_category"] for r in subset)
    other = sum(
        n for category, n in c.items()
        if category not in {
            "successful_landing", "crash_or_out_of_bounds", "timeout"
        }
    )
    lines.append(
        f"| {name} | {len(subset)} | {c['successful_landing']} | "
        f"{c['crash_or_out_of_bounds']} | {c['timeout']} | {other} |"
    )

lines += [
    "",
    "## Interpretation boundaries",
    "",
    "- Categories use the recorded termination reason and explicit landing flag.",
    "- Landing-flag/reason conflicts are reported rather than silently reconciled.",
    "- Reward >= 200 is a separate proxy, not proof of successful landing.",
    "- Leg-contact counts indicate recorded contact only; they do not prove a safe landing.",
    "- Final states and episode lengths support subsequent diagnostics but do not, alone, establish the physical cause of a failure.",
    "- This step classifies outcomes. Trajectory-based causal investigation belongs to Step 2.",
    "",
    "## Model-level breakdown",
    "",
    "| Model | Episodes | Successful landings | Crashes/out of bounds | Timeouts | Mean reward |",
    "|---|---:|---:|---:|---:|---:|",
]
for model in models:
    subset = [r for r in rows if r["model"] == model]
    c = Counter(r["_category"] for r in subset)
    lines.append(
        f"| {model} | {len(subset)} | {c['successful_landing']} | "
        f"{c['crash_or_out_of_bounds']} | {c['timeout']} | "
        f"{fmt(mean([r['_reward'] for r in subset]))} |"
    )

lines += [
    "",
    "## Provenance",
    "",
    "Generated by `experiments/analyze_failure_taxonomy.py` from the existing Phase 02 raw evaluation CSV.",
    "No models were retrained and no evaluation episodes were rerun.",
    "",
]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print("VALIDATION PASSED")
print(f"Input episodes: {len(rows)}")
print(f"Models: {len(models)}")
print(f"Unique records: {len(set(unique_keys))}")
print("\nOverall outcome counts:")
for category in categories:
    if overall[category]:
        print(f"  {category}: {overall[category]} ({100*overall[category]/len(rows):.2f}%)")
print("\nCondition comparison:")
for name, subset in (("fixed", fixed), ("randomized", randomized)):
    c = Counter(r["_category"] for r in subset)
    print(
        f"  {name}: {len(subset)} episodes, "
        f"{c['successful_landing']} landings, "
        f"{c['crash_or_out_of_bounds']} crashes, "
        f"{c['timeout']} timeouts"
    )
print("\nModel breakdown:")
for model in models:
    subset = [r for r in rows if r["model"] == model]
    c = Counter(r["_category"] for r in subset)
    print(
        f"  {model}: landings={c['successful_landing']}, "
        f"crashes={c['crash_or_out_of_bounds']}, "
        f"timeouts={c['timeout']}, "
        f"mean_reward={fmt(mean([r['_reward'] for r in subset]))}"
    )
print(f"\nCreated: {OUT_CSV.relative_to(ROOT)}")
print(f"Created: {OUT_MD.relative_to(ROOT)}")
