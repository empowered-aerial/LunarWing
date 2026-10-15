
import gymnasium as gym
import numpy as np


class RandomizedLunarLander(gym.Wrapper):
    def __init__(
        self,
        seed=None,
        gravity_range=(-11.0, -9.0),
        wind_power_range=(0.0, 15.0),
        turbulence_power_range=(0.0, 1.5),
    ):
        self.rng = np.random.default_rng(seed)
        self.gravity_range = gravity_range
        self.wind_power_range = wind_power_range
        self.turbulence_power_range = turbulence_power_range

        gravity = -10.0
        wind_power = 15.0
        turbulence_power = 1.5

        env = gym.make(
            "LunarLander-v3",
            gravity=gravity,
            enable_wind=True,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )

        super().__init__(env)
        self.current_physics = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        gravity = float(self.rng.uniform(*self.gravity_range))
        wind_power = float(self.rng.uniform(*self.wind_power_range))
        turbulence_power = float(
            self.rng.uniform(*self.turbulence_power_range)
        )

        self.env.close()

        self.env = gym.make(
            "LunarLander-v3",
            gravity=gravity,
            enable_wind=True,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )

        self.current_physics = {
            "gravity": gravity,
            "wind_power": wind_power,
            "turbulence_power": turbulence_power,
        }

        return self.env.reset(seed=seed, options=options)
