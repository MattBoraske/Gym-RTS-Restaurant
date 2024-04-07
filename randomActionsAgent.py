import gym
from gym_environment.TempControlEnv import TemperatureControlEnv

# Define target temperature ranges (start_time, end_time, low, high)
target_temp_ranges = [(0, 25, 18, 20), (26, 50, 22, 24), (51, 75, 16, 18), (76, 100, 20, 22)]

# Create environment
env = TemperatureControlEnv(target_temp_ranges, max_time_steps=target_temp_ranges[-1][1])

for episode in range(1):  # Run 5 episodes
    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        # For testing purposes, using random actions
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward

        # Render environment
        env.render()

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()  # Close environment