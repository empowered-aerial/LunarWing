
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

EVAL_SEEDS = range(100, 110)


MODELS = {
    "fixed_seed123": MODEL_DIR / "ppo_fixed_physics_seed123_100k.zip",
    "randomized_seed123": MODEL_DIR / "ppo_randomized_physics_seed123_100k.zip",
    "fixed_seed2026": MODEL_DIR / "ppo_fixed_physics_seed2026_100k.zip",
    "randomized_seed2026": MODEL_DIR / "ppo_randomized_physics_seed2026_100k.zip",
    "fixed_seed7": MODEL_DIR / "ppo_fixed_physics_seed7_100k.zip",
    "randomized_seed7": MODEL_DIR / "ppo_randomized_physics_seed7_100k.zip",
    "fixed_seed1001": MODEL_DIR / "ppo_fixed_physics_seed1001_100k.zip",
    "randomized_seed1001": MODEL_DIR / "ppo_randomized_physics_seed1001_100k.zip",
}


SCENARIOS = [
    {
        "scenario": "nominal_no_wind",
        "gravity": -10.0,
        "enable_wind": False,
        "wind_power": 15.0,
        "turbulence_power": 1.5,
    },
    {
        "scenario": "strong_gravity_calm",
        "gravity": -11.0,
        "enable_wind": True,
        "wind_power": 0.0,
        "turbulence_power": 0.0,
    },
    {
        "scenario": "strong_gravity_moderate_wind",
        "gravity": -11.0,
        "enable_wind": True,
        "wind_power": 7.5,
        "turbulence_power": 0.75,
    },
    {
        "scenario": "strong_gravity_strong_wind",
        "gravity": -11.0,
        "enable_wind": True,
        "wind_power": 15.0,
        "turbulence_power": 1.5,
    },
    {
        "scenario": "nominal_gravity_turbulence",
        "gravity": -10.0,
        "enable_wind": True,
        "wind_power": 0.0,
        "turbulence_power": 0.75,
    },
    {
        "scenario": "nominal_gravity_windy",
        "gravity": -10.0,
        "enable_wind": True,
        "wind_power": 7.5,
        "turbulence_power": 1.5,
    },
    {
        "scenario": "nominal_gravity_strong_wind",
        "gravity": -10.0,
        "enable_wind": True,
        "wind_power": 15.0,
        "turbulence_power": 0.0,
    },
    {
        "scenario": "weak_gravity_turbulent",
        "gravity": -9.0,
        "enable_wind": True,
        "wind_power": 0.0,
        "turbulence_power": 1.5,
    },
    {
        "scenario": "weak_gravity_moderate_wind",
        "gravity": -9.0,
        "enable_wind": True,
        "wind_power": 7.5,
        "turbulence_power": 0.0,
    },
    {
        "scenario": "weak_gravity_strong_wind",
        "gravity": -9.0,
        "enable_wind": True,
        "wind_power": 15.0,
        "turbulence_power": 0.75,
    },
]



def evaluate_episode(model, scenario, seed):
    env = gym.make(
        "LunarLander-v3",
        gravity=scenario["gravity"],
        enable_wind=scenario["enable_wind"],
        wind_power=scenario["wind_power"],
        turbulence_power=scenario["turbulence_power"],
    )

    try:
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            episode_length += 1

        base_env = env.unwrapped
        left_leg_contact = bool(observation[6] > 0.5)
        right_leg_contact = bool(observation[7] > 0.5)
        crashed = bool(base_env.game_over)

        landed = bool(
            terminated
            and not truncated
            and not crashed
            and not base_env.lander.awake
            and left_leg_contact
            and right_leg_contact
        )

        if truncated:
            termination_reason = "time_limit"
        elif crashed:
            termination_reason = "crash_or_out_of_bounds"
        elif landed:
            termination_reason = "successful_landing"
        else:
            termination_reason = "other_termination"

        return {
            "reward": total_reward,
            "episode_length": episode_length,
            "reward_ge_200": total_reward >= 200,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": termination_reason,
            "landing_success": landed,
            "left_leg_contact": left_leg_contact,
            "right_leg_contact": right_leg_contact,
            "final_horizontal_position": float(observation[0]),
            "final_vertical_position": float(observation[1]),
            "final_horizontal_velocity": float(observation[2]),
            "final_vertical_velocity": float(observation[3]),
            "final_angle": float(observation[4]),
            "final_angular_velocity": float(observation[5]),
        }

    finally:
        env.close()



def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    available_models = {
        name: path
        for name, path in MODELS.items()
        if path.exists()
    }

    missing_models = [
        str(path) for path in MODELS.values()
        if not path.exists()
    ]

    if missing_models:
        print("Models not found; they will be skipped:")
        for path in missing_models:
            print(f"  {path}")

    if not available_models:
        raise FileNotFoundError("No evaluation models were found.")

    records = []

    for model_name, model_path in available_models.items():
        print(f"\nLoading {model_name}: {model_path}")
        model = PPO.load(str(model_path), device="cpu")

        for scenario in SCENARIOS:
            print(f"  Scenario: {scenario['scenario']}")

            for seed in EVAL_SEEDS:
                result = evaluate_episode(model, scenario, seed)

                record = {
                    "model": model_name,
                    "seed": seed,
                    **scenario,
                    **result,
                }

                records.append(record)

                print(
                    f"    Seed {seed}: "
                    f"reward={result['reward']:.2f}, "
                    f"length={result['episode_length']}"
                )

    results = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "generalization_multiseed_evaluation.csv"
    results.to_csv(csv_path, index=False)

    scenario_summary = (
    results.groupby(["model", "scenario"])
    .agg(
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        mean_episode_length=("episode_length", "mean"),
        episodes_reward_ge_200=("reward_ge_200", "sum"),
        landing_successes=("landing_success", "sum"),
        crashes=("termination_reason", lambda s: (s == "crash_or_out_of_bounds").sum()),
        time_limits=("termination_reason", lambda s: (s == "time_limit").sum()),
        other_terminations=("termination_reason", lambda s: (s == "other_termination").sum()),
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
        crashes=("termination_reason", lambda s: (s == "crash_or_out_of_bounds").sum()),
        time_limits=("termination_reason", lambda s: (s == "time_limit").sum()),
        other_terminations=("termination_reason", lambda s: (s == "other_termination").sum()),
        evaluation_episodes=("reward", "size"),
    )
    .reset_index()
)

    summary_path = RESULTS_DIR / "generalization_multiseed_summary.json"
    scenario_path = RESULTS_DIR / "generalization_multiseed_by_scenario.csv"

    summary_path.write_text(
        json.dumps(
            {
                "evaluation_seeds": list(EVAL_SEEDS),
                "scenarios": SCENARIOS,
                "success_threshold_is_proxy": True,
                "model_summary": summary.to_dict(orient="records"),
                "scenario_summary": scenario_summary.to_dict(
                    orient="records"
                ),
                "interpretation": (
                    "The reward >= 200 threshold is a proxy, not a "
                    "verified landing-success metric. The 5k randomized "
                    "pilot is diagnostic only."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scenario_summary.to_csv(scenario_path, index=False)

    print("\nOverall evaluation summary:")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(csv_path)
    print(summary_path)
    print(scenario_path)


if __name__ == "__main__":
    main()


