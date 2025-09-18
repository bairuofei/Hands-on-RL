from gymnasium.envs.registration import register

register(
    id="humannoid_X",
    entry_point="humanoid_v5:HumanoidEnv",
)

import gymnasium as gym
env = gym.make("humannoid_X")

# 打印环境中每个body对应的 qpos 索引和数量
for i, body_name in enumerate(env.model.body_names):
    qpos_start = env.model.body_jntadr[i]  # body 的第一个 joint 在 qpos 的索引
    n_qpos = env.model.body_jntnum[i]      # body 的 joint 数量
    print(f"Body {i}: {body_name}, qpos index start={qpos_start}, num={n_qpos}")


obs, _ = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
