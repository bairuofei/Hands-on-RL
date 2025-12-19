import os
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
import torch.nn as nn

def make_single_env(env_id, seed=0):
    """创建单个环境的工厂函数"""
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)  # gymnasium使用reset(seed=seed)而不是env.seed()
        return env
    return _init

def create_log_dir(base_path="./logs"):
    """创建日志目录"""
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    return base_path

class CustomEvalCallback(EvalCallback):
    """自定义评估回调，增加更详细的日志记录"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_model_path = os.path.join(kwargs.get('best_model_save_path', './logs/models'), 'best_model')
        
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 记录当前训练统计
            if len(self.model.ep_info_buffer) > 0:
                recent_episodes = self.model.ep_info_buffer[-10:]  # 最近10个episode
                train_rewards = [ep['r'] for ep in recent_episodes]
                train_lengths = [ep['l'] for ep in recent_episodes]
                
                self.logger.record("train/mean_reward_recent", np.mean(train_rewards))
                self.logger.record("train/mean_length_recent", np.mean(train_lengths))
                
                print(f"Step {self.n_calls}: Recent train reward: {np.mean(train_rewards):.2f}")
                print(f"Step {self.n_calls}: Best eval reward so far: {self.best_mean_reward:.2f}")
                
        return result
    
    def _on_evaluation_end(self) -> None:
        super()._on_evaluation_end()
        
        # 保存归一化统计量
        if hasattr(self.training_env, 'save') and self.last_mean_reward == self.best_mean_reward:
            norm_stats_path = self.best_model_path + "_norm_stats.pkl"
            self.training_env.save(norm_stats_path)
            print(f"Saved normalization stats to {norm_stats_path}")

def main():
    # 设置随机种子
    seed = 42
    set_random_seed(seed)
    
    # 环境和训练参数
    env_id = "Humanoid-v5"
    n_envs = 8  # 并行环境数量
    total_timesteps = 5_000_000  # 总训练步数
    eval_freq = 10_000  # 每10000步评估一次
    n_eval_episodes = 10  # 每次评估运行10个episode
    
    # 检查Humanoid环境是否可用
    try:
        test_env = gym.make(env_id)
        test_env.close()
        print(f"Environment {env_id} is available")
    except Exception as e:
        print(f"Error creating {env_id}: {e}")
        print("Make sure you have installed MuJoCo: pip install gymnasium[mujoco]")
        return None, 0
    
    # 创建日志目录
    log_dir = create_log_dir()
    
    print(f"Training PPO on {env_id}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Evaluation frequency: {eval_freq:,}")
    print(f"Log directory: {log_dir}")
    print("-" * 50)
    
    # 创建训练环境
    print("Creating training environment...")
    train_env = make_vec_env(
        env_id, 
        n_envs=n_envs, 
        seed=seed,
        vec_env_cls=None  # 使用默认的DummyVecEnv
    )
    
    # 添加归一化
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        training=True,
        gamma=0.99,
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    # 创建评估环境
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        env_id,
        n_envs=1,
        seed=seed + 1000  # 不同的种子
    )
    
    # 评估环境的归一化设置
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # 评估时不归一化奖励
        training=False,     # 不更新统计量
        gamma=0.99,
        clip_obs=10.0
    )
    
    # PPO模型参数
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Actor网络：256-256隐藏层
            vf=[256, 256]   # Critic网络：256-256隐藏层
        ),
        activation_fn=nn.Tanh,  # 激活函数
        ortho_init=False,       # 正交初始化
    )
    
    # 创建PPO模型
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,           # 每次更新收集的步数
        batch_size=64,          # 小批量大小
        n_epochs=10,            # 每次更新的epoch数
        gamma=0.99,             # 折扣因子
        gae_lambda=0.95,        # GAE lambda
        clip_range=0.2,         # PPO剪切范围
        clip_range_vf=None,     # 价值函数剪切
        normalize_advantage=True,
        ent_coef=0.0,          # 熵系数
        vf_coef=0.5,           # 价值函数损失系数
        max_grad_norm=0.5,     # 梯度剪切
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device="auto"
    )
    
    # 设置日志记录
    tb_log_dir = os.path.join(log_dir, "tensorboard")
    model.set_logger(configure(tb_log_dir, ["stdout", "tensorboard"]))
    
    # 创建评估回调
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "models"),
        log_path=os.path.join(log_dir, "evaluations"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
        warn=False
    )
    
    # 打印模型信息
    print("Model configuration:")
    print(f"Policy network: {policy_kwargs['net_arch']}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Batch size: {model.batch_size}")
    print(f"PPO epochs: {model.n_epochs}")
    print(f"Clip range: {model.clip_range}")
    print("-" * 50)
    
    # 开始训练
    try:
        print("Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name="PPO_Humanoid",
            progress_bar=True
        )
        
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # 保存最终模型
        final_model_path = os.path.join(log_dir, "models", "final_model")
        model.save(final_model_path)
        train_env.save(final_model_path + "_norm_stats.pkl")
        print(f"Final model saved to {final_model_path}")
    
    # 最终评估
    print("\nPerforming final evaluation...")
    # 同步归一化统计量
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    print(f"Final evaluation results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 关闭环境
    train_env.close()
    eval_env.close()
    
    print(f"\nTraining logs saved to: {log_dir}")
    print(f"View training progress: tensorboard --logdir {tb_log_dir}")
    
    return model, mean_reward

if __name__ == "__main__":
    # 设置PyTorch线程数（可选，用于性能优化）
    torch.set_num_threads(1)
    
    try:
        model, final_reward = main()
        print(f"\nTraining finished with final reward: {final_reward:.2f}")
    except Exception as e:
        print(f"Training failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Install MuJoCo: pip install gymnasium[mujoco]")
        print("2. Check CUDA availability if using GPU")
        print("3. Reduce n_envs if running out of memory")