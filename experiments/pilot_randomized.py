
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from randomized_env import RandomizedLunarLander


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ppo_randomized_pilot_5000"

SEED = 42
TOTAL_TIMESTEPS = 5_000


def make_env():
    return Monitor(RandomizedLunarLander(seed=SEED))


def main():
    (ROOT / "models").mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env])

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
            seed=SEED,
            device="cpu",
            verbose=1,
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(str(MODEL_PATH))

        print(f"\nPilot model saved to: {MODEL_PATH}.zip")

    finally:
        env.close()

    eval_env = RandomizedLunarLander(seed=100)

    try:
        rewards = []

        for episode in range(3):
            obs, info = eval_env.reset(seed=100 + episode)
            total_reward = 0.0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = (
                    eval_env.step(action)
                )
                total_reward += float(reward)

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            print(
                f"Evaluation episode {episode + 1}: "
                f"reward={total_reward:.2f}, "
                f"physics={eval_env.current_physics}"
            )

        print(f"\nMean pilot evaluation reward: {np.mean(rewards):.2f}")
        print(f"Pilot evaluation std: {np.std(rewards):.2f}")

        assert np.all(np.isfinite(rewards))

        print("PASS: PPO training and evaluation completed.")

    finally:
        eval_env.close()


if __name__ == "__main__":
    main()
