import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
import imageio
import torch
import math
import numpy as np
from tqdm import tqdm
import mujoco

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor



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


def make_env(env_id, rank=0, seed=0):
    """
    env_id: Gymnasium 环境名
    rank: 每个环境编号（用于seed区分）
    seed: 基础随机种子
    """
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        env = CustomRewardWrapper(env)
        env = Monitor(env)
        return env
    return _init

def warm_sin_lr(progress_remaining: float) -> float:
    """
    progress_remaining: 1 -> 0
    假设总共训练T步：
      - 前10% steps: 线性从 1e-5 升到 3e-4 (warm-up)
      - 之后: 按正弦方式从 3e-4 降到 1e-5
    """
    lr_min = 1e-5  
    lr_max = 1e-4
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





if __name__ == '__main__':
    ENV_ID = "Humanoid-v5"
    NUM_ENVS = 16
    SEED = 0

    ## 创建并行环境
    env = SubprocVecEnv([make_env(ENV_ID, i, SEED) for i in range(NUM_ENVS)])
    # env = VecNormalize(env, norm_reward=True)
    env = VecMonitor(env)  # 配合VecMonitor使用
    print(f"env num: {env.num_envs}, obs space: {env.observation_space}, action space: {env.action_space}")

    # # 创建 Ant 环境l
    # env = gym.make('Humanoid-v5')
    # env = CustomRewardWrapper(env)
    # print(f"obs space: {env.observation_space}, action space: {env.action_space}")
    
    # 评估环境
    eval_env = gym.make("Humanoid-v5")   # 你的环境
    eval_env = CustomRewardWrapper(eval_env)

    log_dir = "./tb_log/"
    total_timesteps = 12000000  # 总训练步数

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir+"best_model_ppo2",  # 自动保存最优模型的目录
        log_path=log_dir,                        # 保存评估日志
        eval_freq=10000,                          # 每 1 万步评估一次
        n_eval_episodes=5,                         # 每次评估 5 个 episode
        deterministic=True,                        # 评估时用确定性策略
        render=False
    )



    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]), # 每个隐藏层的神经元数量，也可以写成 [400, 300] 等
        activation_fn=torch.nn.ReLU  # 激活函数，可改为 torch.nn.Tanh
    )

    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=warm_sin_lr,      # 学习率，可以是固定值或调度函数
        n_steps=4096,                    # 每个环境中rollout 采集的步数，相当于 SAC 的 train_freq*gradient_steps
        batch_size=256,                  # 每次梯度更新采样的 batch size
        n_epochs=5,                     # 更新次数=(n_steps * n_envs) / batch_size * n_epochs = 2560次. 一般在200-800次比较合理
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
    model.save("humanoid_ppo2")