from stable_baselines3 import PPO
from scheduling_env import FleetSchedulingEnv
from train_rl_scheduler import evaluate_agent

model = PPO.load("data/rl_scheduler_model")
evaluate_agent(model, n_episodes=10)