import gym
from gym import spaces
import numpy as np

class TemperatureControlEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment simulates a temperature control system where the agent controls
    the power of both heating and cooling units to maintain the temperature within
    specific target ranges by given deadlines.
    """

    def __init__(self, target_temp_ranges, max_time_steps=100):
        super(TemperatureControlEnv, self).__init__()

        # Action space: (cooling unit power, heating unit power)
        # 0: off, 1: low, 2: medium, 3: high
        self.action_space = spaces.MultiDiscrete([4, 4])

        # Observation space includes temperature, external factors, and target range
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([40, 40, 24, 10, 40, 40]),
            dtype=np.float32
        )

        self.current_temperature = 20.0  # Initial temperature
        self.ambient_temperature = 15.0  # Ambient temperature
        self.sunlight_exposure = 0       # Sunlight exposure
        self.number_of_occupants = 0     # Number of occupants

        self.target_temp_ranges = target_temp_ranges
        self.current_target_range = (0, 0, 0, 0)

        self.max_time_steps = max_time_steps
        self.current_time_step = 0

        self.large_negative_reward = -10.0  # Large negative reward for missing deadline

    def step(self, action):
        self.current_time_step += 1

        # Update target temperature range
        self.current_target_range = (0, 0, 0, 0)  # Reset
        for (start, end, low, high) in self.target_temp_ranges:
            if start <= self.current_time_step <= end:
                self.current_target_range = (start, end, low, high)
                break

        self._take_action(action)
        self._simulate_external_factors()

        # Reward calculation
        low, high = self.current_target_range[2], self.current_target_range[3]
        reward = 1.0 if low <= self.current_temperature <= high else -1.0
        if self.current_time_step == self.current_target_range[1] and not (low <= self.current_temperature <= high):
            reward = self.large_negative_reward

        observation = np.array([
            self.current_temperature, 
            self.ambient_temperature, 
            self.sunlight_exposure, 
            self.number_of_occupants, 
            low, high
        ])

        done = self.current_time_step >= self.max_time_steps
        info = {}
        return observation, reward, done, info

    def _take_action(self, action):
        cooling_power, heating_power = action
        temperature_change = heating_power - cooling_power  # Net effect of heating and cooling
        self.current_temperature += temperature_change

    def _simulate_external_factors(self):
        self.ambient_temperature = np.random.normal(15, 5)  # Normal distribution around 15 degrees
        self.sunlight_exposure = np.random.choice([0, 0.5, 1])  # No, partial, or full sunlight
        self.number_of_occupants = np.random.randint(0, 11)  # Up to 10 occupants

    def reset(self):
        # Reset the environment to the initial state
        self.current_temperature = 20.0
        self.ambient_temperature = 15.0
        self.sunlight_exposure = 0
        self.number_of_occupants = 0
        self.current_time_step = 0
        self.current_target_range = (0, 0, 0, 0)
        return np.array([self.current_temperature, self.ambient_temperature, self.sunlight_exposure, self.number_of_occupants, 0, 0])

    def render(self, mode='human'):
        print(f"Time Step: {self.current_time_step}")
        print(f"Current Temperature: {self.current_temperature}")
        print(f"Ambient Temperature: {self.ambient_temperature}")
        print(f"Sunlight Exposure: {self.sunlight_exposure}")
        print(f"Number of Occupants: {self.number_of_occupants}")
        print(f"Current Target Range: {self.current_target_range[2]} - {self.current_target_range[3]}")
        print()

    def close(self):
        pass
