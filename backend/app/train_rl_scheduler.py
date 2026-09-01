from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from scheduling_env import FleetSchedulingEnv


def train_agent():
    env = make_vec_env(lambda: FleetSchedulingEnv(n_engines=10, n_technicians=2, max_steps=100), n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)  # stabilizes training significantly

    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=0.0001,      # lower, more stable
        n_steps=512,               # more experience per update
        batch_size=128,
        ent_coef=0.01,              # encourage a bit more exploration early
        policy_kwargs=dict(net_arch=[128, 128]),  # bigger network for a 10-engine problem
    )

    print("Training PPO agent (this will take longer - real fix, not a shortcut)...")
    model.learn(total_timesteps=500_000)

    model.save("data/rl_scheduler_model")
    env.save("data/vec_normalize.pkl")
    print("\nModel saved.")

    return model, env


def evaluate_agent(model, n_episodes=10):
    # IMPORTANT: evaluate with stochastic actions, not deterministic -
    # for MultiBinary action spaces, "deterministic" thresholds at 0.5,
    # which can behave badly if the policy hasn't fully converged
    env = FleetSchedulingEnv(n_engines=10, n_technicians=2, max_steps=100)

    total_rewards, total_failures, total_services = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=False)  # changed
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated:
                break
        total_rewards.append(ep_reward)
        total_failures.append(info['failures'])
        total_services.append(info['services_performed'])

    print(f"\n{'='*60}")
    print(f"TRAINED AGENT EVALUATION ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Avg reward: {sum(total_rewards)/n_episodes:.1f}  (random baseline: -682)")
    print(f"Avg failures: {sum(total_failures)/n_episodes:.1f}  (random baseline: 5)")
    print(f"Avg services performed: {sum(total_services)/n_episodes:.1f}  (random baseline: 199)")


if __name__ == "__main__":
    model, env = train_agent()
    evaluate_agent(model)