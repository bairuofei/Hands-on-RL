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

 
    
if __name__ == '__main__':
    # 使用可视化界面记录显示SAC测试结果
    # 加载模型
    # model = SAC.load("./tb_log/sac_20250922_160200/best_model.zip")
    # # 创建测试环境
    # env = gym.make("Humanoid-v5", max_episode_steps=100000, render_mode="human")

    use_normalize = True  # 是否使用归一化环境
    ENV_ID = "Humanoid-v5"
    model_dir = "./tb_log/sac_20250923_223358/"
    
    if use_normalize:
        # 创建 demo 环境
        env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="human")])

        # 加载训练时的归一化参数
        import cloudpickle
        with open(model_dir+"vecnormalize_env.pkl", "rb") as f:
            trained_vecnormalize = cloudpickle.load(f)

        # 使用训练环境的均值方差
        env = VecNormalize(env, training=False)
        env.obs_rms = trained_vecnormalize.obs_rms
        env.ret_rms = trained_vecnormalize.ret_rms   
    else: 
        # 创建测试环境
        env = gym.make("Humanoid-v5", render_mode="human")
    
    model = SAC.load(model_dir+"best_model.zip")
    
    obs = env.reset()  # shape = (1, obs_dim)
    for i in range(5):
        obs = env.reset()
        cum_reward = 0
        done = [False]  # VecEnv 返回批量 done
        for _ in tqdm(range(1500)):
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cum_reward += reward[0]  # reward 是批量
            if done[0]:
                print(f"Episode {i+1} 累积奖励: {cum_reward}")
                break
    env.close()