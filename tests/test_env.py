from pettingzoo.mpe import simple_v3
import numpy as np

env = simple_v3.parallel_env()
obs, _ = env.reset(seed=0)
for _ in range(5):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)
print("✅ PettingZoo 并行多智能体 OK")

# 再跑一个 SB3 单环境冒烟
import gymnasium as gym
from stable_baselines3 import PPO
e = gym.make("CartPole-v1")
model = PPO("MlpPolicy", e, n_steps=256, batch_size=64, verbose=0)
model.learn(2000)
print("✅ SB3 训练 OK")

