import gymnasium as gym
from gymnasium.envs.registration import register
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

register(
    id="galaxea_r1Pro",
    entry_point="galaxea_r1Pro:Galaxea_r1Pro",
)




def make_env(env_id, rank=0, seed=0):
    """
    env_id: Gymnasium 环境名
    rank: 每个环境编号（用于seed区分）
    seed: 基础随机种子
    """
    def _init():
        register(
            id="galaxea_r1Pro",
            entry_point="galaxea_r1Pro:Galaxea_r1Pro",
        )
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


 
    
if __name__ == '__main__':
    # 使用可视化界面记录显示SAC测试结果
    # 加载模型
    # model = SAC.load("./tb_log/sac_20250922_160200/best_model.zip")
    # # 创建测试环境
    # env = gym.make("Humanoid-v5", max_episode_steps=100000, render_mode="human")

    use_normalize = True  # 是否使用归一化环境
    ENV_ID = "galaxea_r1Pro"
    model_dir = "./tb_log/sac_20250926_092641/"
    
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
        # env.ret_rms = trained_vecnormalize.ret_rms   
    else: 
        # 创建测试环境
        env = gym.make("Humanoid-v5", render_mode="human")
    
    model = SAC.load(model_dir+"best_model.zip")
    
    obs = env.reset()  # shape = (1, obs_dim)

    # target_list = [np.array([1.5, 0.0, 1.2]), 
    #                np.array([1.5, 1.5, 1.2]),
    #                np.array([0.0, 1.5, 1.2]),
    #                np.array([-1.5, 0.0, 1.2]), 
    #                np.array([0.0, -1.5, 1.2]), 
    #                np.array([1.5, -1.5, 1.2])]
    
    target_list = [np.array([1.7*math.cos(angle), 1.7*math.sin(angle), 1.2]) for angle in np.arange(0, 2*math.pi, math.pi/6)]
    
    for iter_try in range(3):
        idx_tg = 0
        obs = env.reset()
        raw_env = env.envs[0].unwrapped
        raw_env.target_xyz = target_list[idx_tg]
        idx_tg += 1
        # print(raw_env.target_xyz)
        cum_reward = 0
        done = [False]  # VecEnv 返回批量 done
        pbar = tqdm(range(1500))  # 创建 tqdm 对象
        for _ in pbar:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cum_reward += reward[0]  # reward 是批量
            pbar.set_postfix({'reward': reward[0], 'cum_reward': cum_reward})
            if done[0]:
                print(f"Episode {iter_try+1} 累积奖励: {cum_reward}")
                break
            elif reward >= 9.9:
                # update next waypoint
                if idx_tg >= len(target_list):
                    break
                else:
                    raw_env.target_xyz = target_list[idx_tg]
                    idx_tg += 1
                
        env.close()
        
        
    
    # for target_xyz in target_pose:
    #     for i in range(2):
    #         obs = env.reset()
    #         raw_env = env.envs[0].unwrapped
    #         raw_env.target_xyz = target_xyz
    #         # print(raw_env.target_xyz)
    #         cum_reward = 0
    #         done = [False]  # VecEnv 返回批量 done
    #         for _ in tqdm(range(150)):
    #             env.render()
    #             action, _ = model.predict(obs, deterministic=True)
    #             obs, reward, done, info = env.step(action)
    #             cum_reward += reward[0]  # reward 是批量
    #             if done[0]:
    #                 print(f"Episode {i+1} 累积奖励: {cum_reward}")
    #                 break
    # env.close()