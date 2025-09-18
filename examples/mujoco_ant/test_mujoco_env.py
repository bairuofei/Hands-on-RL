import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco

# env = gym.make("Humanoid-v4")  # 或者其他 mujoco env
# obs, info = env.reset()
# action = env.action_space.sample()
# next_obs, reward, terminated, truncated, info = env.step(action)
# done = terminated or truncated
# print(action, reward)


# 窗口可视化结果
env = gym.make("Ant-v5", render_mode="human")  # human 模式会弹出窗口
state, info = env.reset(seed=0)


for j in range(5):
    reward_sum = 0
    state, info = env.reset()
    for i in range(1000):
        action = env.action_space.sample()  # actor选择动作
        next_state, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        env.render()
        if terminated or truncated:
            print("Total reward:", reward_sum)
            break
        else:
            state = next_state


env.close()