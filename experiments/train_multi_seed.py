import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from randomized_env import RandomizedLunarLander


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.0,
    "device": "cpu",
}

TIMESTEPS = 100_000


def train_one(condition, seed, timesteps):
    name = f"ppo_{condition}_physics_seed{seed}_{timesteps // 1000}k"
    model_path = MODEL_DIR / name
    summary_path = RESULTS_DIR / f"{name}_training.json"

    if model_path.with_suffix(".zip").exists() or summary_path.exists():
        raise FileExistsError(
            f"Run already exists: {name}. Refusing to overwrite it."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if condition == "fixed":
        env = gym.make("LunarLander-v3")
    elif condition == "randomized":
        env = RandomizedLunarLander(
            seed=seed,
            gravity_range=(-11.0, -9.0),
            wind_power_range=(0.0, 15.0),
            turbulence_power_range=(0.0, 1.5),
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=PPO_CONFIG["learning_rate"],
            n_steps=PPO_CONFIG["n_steps"],
            batch_size=PPO_CONFIG["batch_size"],
            gamma=PPO_CONFIG["gamma"],
            gae_lambda=PPO_CONFIG["gae_lambda"],
            ent_coef=PPO_CONFIG["ent_coef"],
            seed=seed,
            verbose=1,
            device="cpu",
        )

        print(
            f"Training {condition}, seed {seed}, "
            f"{timesteps:,} timesteps"
        )

        model.learn(total_timesteps=timesteps, progress_bar=False)
        model.save(str(model_path))

        summary = {
            "model": str(model_path.with_suffix(".zip")),
            "condition": condition,
            "training_seed": seed,
            "requested_timesteps": timesteps,
            "actual_timesteps": model.num_timesteps,
            "algorithm": "PPO",
            "environment": (
                {
                    "id": "LunarLander-v3",
                    "gravity": -10.0,
                    "enable_wind": False,
                }
                if condition == "fixed"
                else {
                    "id": "RandomizedLunarLander",
                    "gravity_range": [-11.0, -9.0],
                    "wind_power_range": [0.0, 15.0],
                    "turbulence_power_range": [0.0, 1.5],
                    "enable_wind": True,
                }
            ),
            "ppo": PPO_CONFIG,
            "gymnasium_version": gym.__version__,
            "stable_baselines3_version": stable_baselines3.__version__,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print(f"Saved model: {model_path}.zip")
        print(f"Saved metadata: {summary_path}")

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["fixed", "randomized"],
        default=["fixed", "randomized"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[123, 2026],
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TIMESTEPS,
    )
    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")

    for seed in args.seeds:
        for condition in args.conditions:
            train_one(condition, seed, args.timesteps)


if __name__ == "__main__":
    main()
