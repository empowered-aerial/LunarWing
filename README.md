# LunarWing: Robust Autonomous Lunar Landing Under Uncertain Physics

LunarWing is a reproducible reinforcement-learning research project investigating whether training LunarLander agents under randomized physical conditions improves generalization to previously unseen environment conditions compared with training under fixed physics.

## Research questions

- **RQ1:** Does randomized-physics training improve performance on held-out physics scenarios compared with fixed-physics training?
- **RQ2:** How consistent are the observed differences across training seeds and scenario conditions?

## Approach

The project uses Gymnasium's `LunarLander-v3` environment and Proximal Policy Optimization (PPO) agents. Experiments compare fixed-physics and randomized-physics training, evaluate agents under multiple physics scenarios, and record episode rewards, episode lengths, landing outcomes, and termination information. A separate custom PyTorch episodic actor–critic implementation was explored in Phase 04.

## Repository map

- `experiments/` — training, evaluation, analysis, and environment tests.
- `models/` — saved PPO and custom actor–critic checkpoints.
- `results/` — experiment records, raw evaluation outputs, statistical summaries, and phase records.
- `results/final_tables/` — curated validated summary tables.
- `results/final_figures/` — figures summarizing model and scenario comparisons.
- `results/phase04_actor_critic/` — Phase 04 training log, curves, and record.
- `logs/` — experiment logs, where present.
- `src/` — supporting source code, where present.

## Environment setup

Use Python 3.12 and a virtual environment. The commands below are for Windows PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dependency snapshot was generated from the project's existing virtual environment. For a clean reproduction, verify that all pinned package versions install successfully on your operating system. LunarLander also requires compatible Box2D dependencies.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q experiments/test_randomized_env.py
```

## Reproduce experiments

Training and evaluation scripts are in `experiments/`. Inspect each script's argument parser and configuration before running it. For example, use the following to inspect the available options:

```powershell
.\.venv\Scripts\python.exe experiments\train_multi_seed.py --help
.\.venv\Scripts\python.exe experiments\evaluate_generalization.py --help
```

Run the intended training script before evaluation if you are reproducing an experiment from scratch. Existing checkpoints and recorded results are supplied to support inspection without retraining. Exact commands, seed settings, scenario definitions, and output paths should be taken from the corresponding experiment scripts and records.

## Results and figures

Start with:

- `results/EXPERIMENT_RECORD.md`
- `results/PHASE_02_STATISTICAL_RECORD.md`
- `results/phase02_statistical_summary.json`
- `results/phase03_failure_taxonomy.md`
- `results/final_tables/model_summary.csv`
- `results/final_tables/paired_reward_confidence_intervals.csv`
- `results/final_tables/scenario_summary.csv`
- `results/final_figures/model_mean_rewards.png`
- `results/final_figures/paired_reward_differences.png`
- `results/final_figures/scenario_performance.png`
- `results/phase04_actor_critic/PHASE04_RECORD.md`

Files whose names contain `before_` are historical intermediate outputs, not the preferred final results. Prefer validated files explicitly marked `validated` and the curated `final_tables/` directory. Consult the experiment records for provenance and metric definitions.

## Interpretation and limitations

Reward is not identical to a successful landing. Interpret reward alongside explicit landing-success, crash, and termination metrics. Generalization results apply to the tested physics scenarios and seeds; they do not establish robustness to every possible lunar-landing condition.

Phase 04's custom actor–critic run is documented as implementation-complete, not learning-successful. Its results should not be presented as evidence of a successful landing policy.

## Reproducibility notes

Random seeds, training budgets, scenario settings, dependency versions, and evaluation procedures affect results. Existing outputs support auditability, but a clean-environment rerun is needed to independently confirm end-to-end reproducibility. Checkpoint files are provided for evaluation; retraining may produce different results across software versions and hardware.

## Citation

See `CITATION.cff` for the repository citation metadata. Repository: https://github.com/empowered-aerial/LunarWing
Current artifact version: 1.0.0. This version identifies the initial packaged research artifact; independent reproduction has not yet been verified.
