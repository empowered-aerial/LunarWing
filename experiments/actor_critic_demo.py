
import csv
import json
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "phase04_actor_critic"
MODEL_DIR = ROOT / "models"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPISODES = 100
GAMMA = 0.99
LEARNING_RATE = 0.0003
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
MAX_GRAD_NORM = 0.5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    env = gym.make("LunarLander-v3")
except gym.error.Error:
    env = gym.make("LunarLander-v2")

env.action_space.seed(SEED)


class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, observations):
        features = self.shared(observations)
        action_logits = self.actor(features)
        state_values = self.critic(features).squeeze(-1)
        return action_logits, state_values

    def act(self, observation, deterministic=False):
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)

        logits, value = self(observation)
        distribution = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = distribution.sample()

        log_probability = distribution.log_prob(action)
        entropy = distribution.entropy()

        return (
            int(action.item()),
            log_probability.squeeze(0),
            value.squeeze(0),
            entropy.squeeze(0)
        )


def discounted_returns(rewards, gamma):
    returns = []
    running_return = 0.0

    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)

    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=DEVICE)


def evaluate(model, episodes=10):
    scores = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=10000 + episode)
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _, _, _ = model.act(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        scores.append(total_reward)

    return {
        "mean_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "episode_rewards": [float(score) for score in scores]
    }


model = ActorCritic(
    env.observation_space.shape[0],
    env.action_space.n
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

initial_evaluation = evaluate(model)
training_records = []
gradient_norms = []

for episode in range(1, EPISODES + 1):
    observation, _ = env.reset(seed=SEED + episode)

    log_probabilities = []
    values = []
    entropies = []
    rewards = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, log_probability, value, entropy = model.act(observation)

        next_observation, reward, terminated, truncated, _ = env.step(action)

        log_probabilities.append(log_probability)
        values.append(value)
        entropies.append(entropy)
        rewards.append(float(reward))

        observation = next_observation

    returns = discounted_returns(rewards, GAMMA)
    log_probabilities = torch.stack(log_probabilities)
    values = torch.stack(values)
    entropies = torch.stack(entropies)

    advantages = returns - values.detach()

    policy_loss = -(log_probabilities * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_bonus = entropies.mean()

    loss = (
        policy_loss
        + VALUE_COEFFICIENT * value_loss
        - ENTROPY_COEFFICIENT * entropy_bonus
    )

    optimizer.zero_grad()
    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), MAX_GRAD_NORM
    )
    optimizer.step()

    gradient_norms.append(float(gradient_norm))

    training_records.append({
        "episode": episode,
        "reward": sum(rewards),
        "length": len(rewards),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy_bonus.detach()),
        "total_loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm)
    })

    if episode % 10 == 0:
        recent = [record["reward"] for record in training_records[-10:]]
        print(
            f"Episode {episode:3d}/{EPISODES} | "
            f"Mean reward (last 10): {np.mean(recent):8.2f} | "
            f"Policy loss: {policy_loss.item():8.3f} | "
            f"Value loss: {value_loss.item():8.3f}"
        )

final_evaluation = evaluate(model)

checkpoint_path = MODEL_DIR / "phase04_actor_critic.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "observation_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.n,
    "seed": SEED,
    "gamma": GAMMA,
    "learning_rate": LEARNING_RATE,
    "episodes": EPISODES
}, checkpoint_path)

csv_path = RESULT_DIR / "training_log.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=training_records[0].keys())
    writer.writeheader()
    writer.writerows(training_records)

episode_numbers = [record["episode"] for record in training_records]
rewards = [record["reward"] for record in training_records]
policy_losses = [record["policy_loss"] for record in training_records]
value_losses = [record["value_loss"] for record in training_records]

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

axes[0].plot(episode_numbers, rewards, alpha=0.45, label="Episode reward")
if len(rewards) >= 10:
    moving_average = np.convolve(rewards, np.ones(10) / 10, mode="valid")
    axes[0].plot(
        episode_numbers[9:], moving_average, label="10-episode moving average"
    )
axes[0].set_title("LunarLander Actor-Critic Training Rewards")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].legend()

axes[1].plot(episode_numbers, policy_losses, label="Policy loss")
axes[1].set_title("Policy Loss")
axes[1].set_xlabel("Episode")
axes[1].legend()

axes[2].plot(episode_numbers, value_losses, label="Value loss")
axes[2].set_title("Critic Value Loss")
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("MSE")
axes[2].legend()

fig.tight_layout()
plot_path = RESULT_DIR / "training_curves.png"
fig.savefig(plot_path, dpi=160)
plt.close(fig)

summary = {
    "environment": env.spec.id,
    "algorithm": "Vanilla episodic actor-critic",
    "episodes": EPISODES,
    "seed": SEED,
    "device": str(DEVICE),
    "initial_evaluation": initial_evaluation,
    "final_evaluation": final_evaluation,
    "last_20_training_mean_reward": float(
        np.mean([record["reward"] for record in training_records[-20:]])
    ),
    "mean_gradient_norm": float(np.mean(gradient_norms)),
    "checkpoint": str(checkpoint_path),
    "training_log": str(csv_path),
    "plot": str(plot_path),
    "note": (
        "The critic uses Monte Carlo discounted returns. "
        "This is a standalone educational actor-critic, not PPO."
    )
}

summary_path = RESULT_DIR / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n========== PHASE 04 RESULTS ==========")
print(json.dumps(summary, indent=2))
print("\nSaved training plot:", plot_path)
print("Saved checkpoint:", checkpoint_path)
print("Saved training log:", csv_path)
print("Saved summary:", summary_path)

env.close()
