# %%
import gymnasium as gym
from gymnasium.envs.registration import register
import mujoco
from tqdm import tqdm
import torch
import math

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.monitor import Monitor

import os
from datetime import datetime

# 保存整个 VecNormalize 对象
import cloudpickle

# %%
register(
    id="galaxea_r1Pro",
    entry_point="galaxea_r1Pro:Galaxea_r1Pro",
)


# %%
def make_env(env_id, rank=0, seed=0, max_episode_steps=1000):
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
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init




# %%
class RewardInfoCallback(BaseCallback):
    """
    在 TensorBoard 中记录 info 字典中的各个 reward 项。
    支持 vectorized environments。
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # 对 vectorized env 平均 reward
        reward_sums = {}
        for info in infos:
            for key, value in info.items():
                if key.startswith("reward_"):
                    reward_sums[key] = reward_sums.get(key, 0.0) + float(value)

        for key, total in reward_sums.items():
            mean_value = total / len(infos)
            self.logger.record(f"reward/{key}", mean_value)

        return True
    


def warm_sin_lr(progress_remaining: float) -> float:
    """
    progress_remaining: 1 -> 0
    假设总共训练T步：RewardInfoCallback
      - 前10% steps: 线性从 1e-5 升到 3e-4 (warm-up)
      - 之后: 按正弦方式从 3e-4 降到 1e-5
    """
    lr_min = 1e-4   
    lr_max = 1e-3
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


class SyncVecNormalizeEvalCallback(EvalCallback):
    def __init__(self, *args, train_env=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_env = train_env

    def _on_step(self) -> bool:
        # 每次评估前同步归一化参数
        old_best = self.best_mean_reward
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 对vectorized environment同样适用
            if (self.train_env is not None and hasattr(self.train_env, 'obs_rms') and hasattr(self.eval_env, 'obs_rms')):
                # 直接复制整个obs_rms对象
                self.eval_env.obs_rms = self.train_env.obs_rms
                print(f"Synced vectorized env normalization at step {self.n_calls}")
            
        result = super()._on_step()
        # 如果 best_mean_reward 变了说明更新了
        if self.best_mean_reward > old_best:
            # 保存最新的归一化参数
            with open(best_model_dir + "vecnormalize_env.pkl", "wb") as f:
                cloudpickle.dump(self.train_env, f)
        return result

if __name__ == '__main__':
    time_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")   # 例如 20250922_113045

    log_dir = "./tb_log/"
    best_model_dir = log_dir+"sac_"+time_prefix+"/"
    
    # Environment
    ENV_ID = "galaxea_r1Pro"
    NUM_ENVS = 12
    SEED = 0
    EVAL_SEED = 100
    max_episode_steps=1500
    
    # Training
    total_timesteps = 4800000  # 总训练步数
    buffer_size = 40000  # 经验回放缓冲区大小
    
    # EWvalation
    eval_freq = 10000      # 每多少步评估一次
    n_eval_episodes = 5    # 每次评估多少个回合


    ## 创建并行环境
    train_env = SubprocVecEnv([make_env(ENV_ID, i, SEED, max_episode_steps) for i in range(NUM_ENVS)])
    train_env = VecMonitor(train_env)  # 配合VecMonitor使用
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)  # 训练环境归一化
    print(f"env num: {train_env.num_envs}, obs space: {train_env.observation_space}, action space: {train_env.action_space}")
    print("Max episode steps:", train_env.get_attr('spec')[0].max_episode_steps)


    # 评估环境
    eval_env = SubprocVecEnv([make_env(ENV_ID, 0, EVAL_SEED, max_episode_steps) for _ in range(1)])   # 你的环境
    eval_env = VecMonitor(eval_env)  # 配合VecMonitor使用
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # 不更新统计量


    eval_callback = SyncVecNormalizeEvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,  # 自动保存最优模型的目录
        log_path=log_dir,                        # 保存评估日志
        eval_freq=eval_freq,                          # 每 1 万步评估一次
        n_eval_episodes=n_eval_episodes,                         # 每次评估 5 个 episode
        deterministic=True,                        # 评估时用确定性策略
        render=False, 
        train_env=train_env  # 传入训练环境
    )
    
    reward_info_cb = RewardInfoCallback()

    # 自定义SAC网络结构
    # obs space: (706,), action space: (24,)
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]), # 每个隐藏层的神经元数量，也可以写成 [400, 300] 等
        activation_fn=torch.nn.ReLU  # 激活函数，可改为 torch.nn.Tanh
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=warm_sin_lr,
        buffer_size=buffer_size,      # 经验回放缓冲区大小. 这个参数PPO没有
        learning_starts=1000, 
        batch_size=256,             # 默认256
        tau=0.005,                  # 软更新系数
        gamma=0.99,                 # 折扣因子
        train_freq=1,               # 每步都训练，采集多少个环境步的数据后训练一次
        gradient_steps=1,           # 对replayBuffer中读取到的batch，进行多少次梯度下降更新
        tensorboard_log=log_dir,   # 日志目录
        policy_kwargs=policy_kwargs,  # 将自定义结构传进去
    )

    # 训练模型, total_timesteps自行调整
    model.learn(total_timesteps=total_timesteps, 
                tb_log_name="sac_"+time_prefix,
                progress_bar=True,
                callback=[eval_callback, reward_info_cb])
    # 保存模型
    save_path = os.path.join(best_model_dir, "final_model")
    model.save(save_path)

    train_env.close()
        
    with open(best_model_dir + "vecnormalize_env.pkl", "wb") as f:
        cloudpickle.dump(train_env, f)
