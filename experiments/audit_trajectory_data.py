import csv
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent.parent
results = root / "results"
experiments = root / "experiments"
raw = results / "generalization_multiseed_evaluation.csv"

print("=" * 75)
print("PHASE 03 STEP 2 — TRAJECTORY DATA AUDIT")
print("=" * 75)

print("\n1. RAW EVALUATION DATA")
if not raw.exists():
    print(f"ERROR: Expected input not found: {raw}")
    print("Available CSV files:")
    for p in sorted(results.glob("*.csv")):
        print(f"  {p.name}")
else:
    with raw.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    print(f"Input: {raw.relative_to(root)}")
    print(f"Rows: {len(rows)}")
    print("Columns:")
    for field in fields:
        print(f"  {field}")

    for field in fields:
        if any(word in field.lower() for word in (
            "reason", "terminat", "truncat", "landing", "contact"
        )):
            counts = Counter(str(r.get(field, "")) for r in rows)
            print(f"\nValues for {field}:")
            for value, count in counts.most_common():
                print(f"  {value!r}: {count}")

    print("\nPotential per-step fields:")
    matches = [
        f for f in fields
        if any(word in f.lower() for word in (
            "trajectory", "trace", "step_states", "state_history",
            "observations", "positions", "actions", "state_sequence"
        ))
    ]
    print(matches if matches else "No obvious trajectory/history columns found.")

print("\n2. EXISTING RESULT FILES")
for folder in (results, experiments):
    print(f"\n{folder.relative_to(root)}/")
    for p in sorted(folder.rglob("*")):
        if p.is_file() and (
            any(word in p.name.lower() for word in (
                "generalization", "evaluation", "trajectory", "trace",
                "episode", "environment", "train", "evaluate"
            ))
        ):
            print(f"  {p.relative_to(root)} ({p.stat().st_size} bytes)")

print("\n3. SOURCE CODE REFERENCES")
keywords = (
    "termination_reason", "final_horizontal_position",
    "final_vertical_position", "final_horizontal_velocity",
    "final_vertical_velocity", "terminated", "truncated",
    "episode_length", "record", "trajectory", "Monitor",
    "TimeLimit", "max_episode_steps"
)
for folder in (experiments, root / "src"):
    if not folder.exists():
        continue
    for p in sorted(folder.rglob("*.py")):
        if any(part.startswith(".") or part == "__pycache__"
               for part in p.parts):
            continue
        try:
            lines = p.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError):
            continue
        hits = [
            (i, line.strip())
            for i, line in enumerate(lines, 1)
            if any(k.lower() in line.lower() for k in keywords)
        ]
        if hits:
            print(f"\n--- {p.relative_to(root)} ---")
            for i, line in hits[:35]:
                print(f"{i}: {line[:220]}")
            if len(hits) > 35:
                print(f"... {len(hits) - 35} additional matching lines")

print("\nAUDIT COMPLETE")
