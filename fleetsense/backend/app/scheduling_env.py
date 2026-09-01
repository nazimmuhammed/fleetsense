"""
FleetSense - RL Scheduling Environment
==========================================
A custom Gymnasium environment simulating fleet maintenance scheduling.

Setup: N engines exist simultaneously, each with a predicted RUL (from our
LSTM) that decreases over time. Only K technicians are available per time
step. The agent must decide which engines to service each step.

Reward logic:
- Big penalty if an engine's RUL hits 0 (unplanned failure - the worst outcome)
- Small penalty for servicing an engine too early (wasted technician capacity
  on a healthy engine)
- Reward for servicing an engine that was genuinely at risk (low RUL)
- Small reward per step for keeping the whole fleet healthy
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FleetSchedulingEnv(gym.Env):
    def __init__(self, n_engines=10, n_technicians=2, max_steps=100):
        super().__init__()
        self.n_engines = n_engines
        self.n_technicians = n_technicians
        self.max_steps = max_steps

        # Action: for each engine, decide 0 (don't service) or 1 (service this step)
        # Total technicians used must not exceed n_technicians (enforced in step())
        self.action_space = spaces.MultiBinary(n_engines)

        # Observation: current RUL of each engine, normalized 0-1
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_engines,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize each engine with a random starting RUL (simulating a fleet
        # at different points in their operational life)
        self.rul = np.random.randint(20, 150, size=self.n_engines).astype(np.float32)
        self.max_rul = self.rul.copy()  # for normalization
        self.step_count = 0
        self.failures = 0
        self.services_performed = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return (self.rul / self.max_rul).astype(np.float32)

    def step(self, action):
        action = np.array(action)

        # Enforce technician capacity constraint: if agent tries to service
        # more engines than technicians available, only the first K (by
        # action order) are actually serviced - agent must learn to prioritize
        service_indices = np.where(action == 1)[0]
        if len(service_indices) > self.n_technicians:
            service_indices = service_indices[:self.n_technicians]

        reward = 0
        for i in range(self.n_engines):
            if i in service_indices:
                # Servicing resets RUL to a fresh value (simulating maintenance)
                if self.rul[i] < 30:
                    reward += 10  # good: serviced a genuinely at-risk engine
                else:
                    reward -= 3   # wasteful: serviced a healthy engine, wasted capacity
                self.rul[i] = self.max_rul[i]
                self.services_performed += 1
            else:
                self.rul[i] -= 1  # engine continues degrading
                if self.rul[i] <= 0:
                    reward -= 50  # bad: unplanned failure, the worst outcome
                    self.failures += 1
                    self.rul[i] = self.max_rul[i]  # "replace" and continue simulation

        reward += 1  # small reward per step for keeping simulation running (fleet operational)

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False

        info = {"failures": self.failures, "services_performed": self.services_performed}
        return self._get_obs(), reward, terminated, truncated, info


if __name__ == "__main__":
    # Quick sanity test: random policy baseline
    env = FleetSchedulingEnv(n_engines=10, n_technicians=2, max_steps=100)
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    print(f"Random policy baseline - Total reward: {total_reward}")
    print(f"Failures: {info['failures']}, Services performed: {info['services_performed']}")