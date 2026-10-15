
import json
from pathlib import Path

import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SOURCE_MODEL = MODEL_DIR / "ppo_fixed_physics.zip"
OUTPUT_MODEL = MODEL_DIR / "ppo_fixed_physics_300k"
EVAL_SEEDS = range(100, 110)
ADDITIONAL_TIMESTEPS = 200_000


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Missing model: {SOURCE_MODEL}")

    env = gym.make("LunarLander-v3")
    model = PPO.load(str(SOURCE_MODEL), env=env, device="cpu")

    print("Continuing training from the existing 100k model.")
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        reset_num_timesteps=False,
    )

    model.save(str(OUTPUT_MODEL))
    env.close()

    eval_env = gym.make("LunarLander-v3")
    records = []

    for seed in EVAL_SEEDS:
        observation, info = eval_env.reset(seed=seed)
        total_reward = 0.0
        length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = (
                eval_env.step(action)
            )
            total_reward += float(reward)
            length += 1

        records.append(
            {
                "seed": seed,
                "reward": total_reward,
                "episode_length": length,
                "reward_ge_200": total_reward >= 200,
            }
        )
        print(f"Seed {seed}: reward={total_reward:.2f}, length={length}")

    eval_env.close()

    results = pd.DataFrame(records)
    results.to_csv(
        RESULTS_DIR / "ppo_fixed_physics_300k.csv",
        index=False,
    )

    summary = {
        "environment": "LunarLander-v3",
        "algorithm": "PPO",
        "nominal_total_timesteps": 300_000,
        "additional_timesteps": ADDITIONAL_TIMESTEPS,
        "evaluation_seeds": list(EVAL_SEEDS),
        "mean_reward": float(results["reward"].mean()),
        "std_reward": float(results["reward"].std(ddof=1)),
        "episodes_reward_ge_200": int(results["reward_ge_200"].sum()),
        "evaluation_episodes": len(results),
    }

    with open(
        RESULTS_DIR / "ppo_fixed_physics_300k_summary.json",
        "w",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nFinal summary:")
    print(json.dumps(summary, indent=2))
    print("\nModel saved to:", OUTPUT_MODEL)


if __name__ == "__main__":
    main()
