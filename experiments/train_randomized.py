
import json
from pathlib import Path
from datetime import datetime, timezone

import gymnasium as gym
from stable_baselines3 import PPO

from randomized_env import RandomizedLunarLander


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = MODEL_DIR / "ppo_randomized_100k"
SUMMARY_PATH = RESULTS_DIR / "ppo_randomized_100k_training.json"

TOTAL_TIMESTEPS = 100_000
TRAIN_SEED = 42


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env = RandomizedLunarLander(seed=TRAIN_SEED)

    try:
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

        print(f"Training randomized LunarLander for {TOTAL_TIMESTEPS:,} timesteps.")

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=False,
        )

        model.save(str(MODEL_PATH))

        summary = {
            "model": str(MODEL_PATH.with_suffix(".zip")),
            "requested_timesteps": TOTAL_TIMESTEPS,
            "actual_timesteps": model.num_timesteps,
            "training_seed": TRAIN_SEED,
            "environment": "RandomizedLunarLander",
            "ppo": {
                "learning_rate": 3e-4,
                "n_steps": 1024,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.0,
            },
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

        print("\nTraining completed.")
        print(f"Actual timesteps: {model.num_timesteps:,}")
        print(f"Model saved: {MODEL_PATH}.zip")
        print(f"Training record saved: {SUMMARY_PATH}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
