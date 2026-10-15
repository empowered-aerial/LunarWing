
import numpy as np
from randomized_env import RandomizedLunarLander


def run_sequence(seed):
    env = RandomizedLunarLander(seed=seed)
    conditions = []

    try:
        for episode in range(5):
            observation, info = env.reset()
            physics = env.current_physics.copy()
            underlying = env.unwrapped

            assert np.isclose(
                underlying.world.gravity.y,
                physics["gravity"],
            )
            assert np.isclose(
                underlying.wind_power,
                physics["wind_power"],
            )
            assert np.isclose(
                underlying.turbulence_power,
                physics["turbulence_power"],
            )

            for _ in range(25):
                previous_physics = env.current_physics.copy()
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                assert env.current_physics == previous_physics
                assert np.all(np.isfinite(observation))
                assert np.isfinite(reward)

                if terminated or truncated:
                    break

            conditions.append(physics)

        return conditions

    finally:
        env.close()


def main():
    first = run_sequence(123)
    second = run_sequence(123)

    assert first == second, "Physics sampling is not reproducible."

    assert len({
        tuple(sorted(item.items()))
        for item in first
    }) > 1

    print("PASS: Sampled physics matches actual environment physics.")
    print("PASS: Physics settings remain constant within each episode.")
    print("PASS: Identical seeds reproduce the physics sequence.")
    print("PASS: Different episodes receive randomized physics.")


if __name__ == "__main__":
    main()
