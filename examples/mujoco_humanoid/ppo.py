# %%
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
import imageio
import torch
import math
import numpy as np
from tqdm import tqdm
import mujoco


# %%
def quaternion_angle_error(q1, q2):
    # q1, q2 shape: (4,), format: [w, x, y, z]
    dot = np.abs(np.dot(q1, q2))  # 绝对值，避免 2π 距离问题
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2 * np.arccos(dot)
    return abs(angle)


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.w_upward = 1
        self.w_upfoot = 1
        self.w_uphead = 1

        self.target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # ------------------------
        # 1. 躯干挺直 torso orientation (w, x, y, z) 在 obs[1:5]
        torso_ori = obs[1:5]
        ori_error = quaternion_angle_error(torso_ori, self.target_orientation)
        rew_upward = np.clip(self.w_upward * ori_error, 0, 5)
        
        # 2. 保持脑袋高度
        rew_uphead = np.clip(self.w_uphead * (self.env.unwrapped.data.xipos[1][2] - 0.9), -5, 5)
        # 3. 鼓励抬脚
        footreward = 0
        if  0.2 <= self.env.unwrapped.data.xipos[6][2] <= 0.6:
            footreward += 1
        if  0.2 <= self.env.unwrapped.data.xipos[9][2] <= 0.6:
            footreward += 1
        rew_foot = self.w_upfoot * footreward # [0, 2]

        new_reward = reward + rew_foot + rew_uphead - rew_upward
        return obs, new_reward, terminated, truncated, info


# %%
# 创建 Ant 环境l
env = gym.make('Humanoid-v5')
env = CustomRewardWrapper(env)
print(f"obs space: {env.observation_space}, action space: {env.action_space}")


# 评估环境
eval_env = gym.make("Humanoid-v5")   # 你的环境
eval_env = CustomRewardWrapper(eval_env)

log_dir = "./tb_log/"
total_timesteps = 4800000  # 总训练步数

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir+"best_model_ppo",  # 自动保存最优模型的目录
    log_path=log_dir,                        # 保存评估日志
    eval_freq=10000,                          # 每 1 万步评估一次
    n_eval_episodes=5,                         # 每次评估 5 个 episode
    deterministic=True,                        # 评估时用确定性策略
    render=False
)

# %%
policy_kwargs = dict(
    net_arch=dict(pi=[64], qf=[64]), # 每个隐藏层的神经元数量，也可以写成 [400, 300] 等
    activation_fn=torch.nn.ReLU  # 激活函数，可改为 torch.nn.Tanh
)

def warm_sin_lr(progress_remaining: float) -> float:
    """
    progress_remaining: 1 -> 0
    假设总共训练T步：
      - 前10% steps: 线性从 1e-5 升到 3e-4 (warm-up)
      - 之后: 按正弦方式从 3e-4 降到 1e-5
    """
    lr_min = 1e-3  
    lr_max = 5e-3
    warm_ratio = 0.01  # 10% warm-up

    # progress_remaining=1 -> step=0; progress_remaining=0 -> step=end
    progress_done = 1.0 - progress_remaining

    if progress_done < warm_ratio:
        # warm-up: 线性上升
        return lr_min + (lr_max - lr_min) * (progress_done / warm_ratio)
    else:
        # sin下降：这里重新归一化到[0,1]
        x = (progress_done - warm_ratio) / (1 - warm_ratio)
        return lr_min + (lr_max - lr_min) * math.sin((1 - x) * math.pi / 2)



model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=warm_sin_lr,      # 学习率，可以是固定值或调度函数
    n_steps=4096,                    # PPO 每次 rollout 采集的步数，相当于 SAC 的 train_freq*gradient_steps
    batch_size=256,                  # 每次梯度更新采样的 batch size
    n_epochs=20,
    gamma=0.99,                      # 折扣因子
    gae_lambda=0.95,                 # GAE 参数
    clip_range=0.2,
    ent_coef=0.02,                     # 熵正则系数，控制探索
    vf_coef=0.5,                      # value loss 权重
    max_grad_norm=0.5,               # 梯度裁剪
    tensorboard_log=log_dir,      # TensorBoard 日志目录
    policy_kwargs=policy_kwargs       # 自定义网络结构
)

# 训练模型, total_timesteps自行调整
model.learn(total_timesteps=total_timesteps, 
            tb_log_name="ppo", 
            progress_bar=True,
            callback=eval_callback)
# 保存模型
model.save("humanoid_ppo")

# %% [markdown]
# ### 测试模型效果

# %%
print(gym.__file__)

# %%
# 使用可视化界面记录显示SAC测试结果
# 加载模型
model = PPO.load("./humanoid_ppo.zip")
# 创建测试环境
env = gym.make("Humanoid-v5", render_mode="human")

for i in range(5):
    # 测试模型
    state, info = env.reset()
    cum_reward = 0
    for _ in tqdm(range(1500)):
        env.render()
        action, _ = model.predict(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        if terminated or truncated:
            print("累积奖励: ", cum_reward)
            break
            
        state = next_state

env.close()

# %% [markdown]
# ### 测试代码

# %%
!tensorboard --logdir ./ppo_ant_tb/
!tensorboard --logdir ./tb_log/
# then 然后浏览器打开 http://localhost:6006


