from stable_baselines3.common.env_checker import check_env
from RTS_Restaurant_Env import RTS_Restaurant_Env
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, A2C, DQN

TOTAL_WORKERS = 5
TASKS = [
    {"task_type": 'FOH', "yield_per_worker": 1, "item_ordering_weight": 0, 'storage_capacity': 0},
    {"task_type": 'BOH', "yield_per_worker": 0.2, "item_ordering_weight": 2, 'storage_capacity': 50},
    {"task_type": 'BOH', "yield_per_worker": 0.4, "item_ordering_weight": 3, 'storage_capacity': 50},
    {"task_type": 'BOH', "yield_per_worker": 0.6, "item_ordering_weight": 5, 'storage_capacity': 50},
]
MAX_NEW_CUSTOMERS = 3
MAX_TOTAL_CUSTOMERS = 100

env = RTS_Restaurant_Env(TOTAL_WORKERS, TASKS, MAX_NEW_CUSTOMERS, MAX_TOTAL_CUSTOMERS)
#print(env.observation_space)
env = FlattenObservation(env)
#print()
#print(env.observation_space)
check_env(env)

model = PPO("MlpPolicy", env).learn(total_timesteps=1000)
model.save("trained_agents/ppo_restaurant_worker_scheduler")
