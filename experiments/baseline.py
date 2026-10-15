import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TOTAL_TIMESTEPS = 100_000
TRAIN_SEED = 42
EVAL_SEEDS = list(range(100, 110))


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make("LunarLander-v3")
    env.reset(seed=TRAIN_SEED)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        seed=TRAIN_SEED,
        verbose=1,
        device="cpu",
    )

    print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps.")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model_path = MODEL_DIR / "ppo_fixed_physics"
    model.save(str(model_path))
    env.close()

    eval_env = gym.make("LunarLander-v3")
    records = []

    for seed in EVAL_SEEDS:
        observation, info = eval_env.reset(seed=seed)
        total_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = (
                eval_env.step(action)
            )
            total_reward += float(reward)
            episode_length += 1

        records.append(
            {
                "seed": seed,
                "reward": total_reward,
                "episode_length": episode_length,
                "reward_ge_200": total_reward >= 200,
            }
        )

        print(
            f"Seed {seed}: reward={total_reward:.2f}, "
            f"length={episode_length}"
        )

    eval_env.close()

    results = pd.DataFrame(records)
    results.to_csv(RESULTS_DIR / "ppo_fixed_physics.csv", index=False)

    summary = {
        "environment": "LunarLander-v3",
        "algorithm": "PPO",
        "training_seed": TRAIN_SEED,
        "evaluation_seeds": EVAL_SEEDS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "mean_reward": float(results["reward"].mean()),
        "std_reward": float(results["reward"].std(ddof=1)),
        "mean_episode_length": float(results["episode_length"].mean()),
        "episodes_reward_ge_200": int(results["reward_ge_200"].sum()),
        "evaluation_episodes": len(results),
        "success_threshold_is_proxy": True,
    }

    with open(RESULTS_DIR / "ppo_fixed_physics_summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print("\nEvaluation summary:")
    print(json.dumps(summary, indent=2))
    print("\nSaved model:", model_path)
    print("Saved results:", RESULTS_DIR)


if __name__ == "__main__":
    main()
