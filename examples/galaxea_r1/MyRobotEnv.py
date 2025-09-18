import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces

class MyRobotEnv(MujocoEnv):
    def __init__(self, xml_file="./galaxea_r1pro/r1_pro.xml"):
        # 机器人 xml 文件路径
        full_path = os.path.join(os.path.dirname(__file__), xml_file)

        # 调用父类构造函数
        super().__init__(
            model_path=full_path,
            frame_skip=5,           # 每次step模拟5帧，可根据需要调整
            observation_space=None, # 我们在 _get_obs 中自定义
            default_camera_config={}
        )

        # 自定义 observation_space，这里示例用位置+速度
        obs_size = self.data.qpos.size + self.data.qvel.size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # 如果有连续动作空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float64
        )

    def _get_obs(self):
        # 返回观测值：位置+速度
        return np.concatenate([self.data.qpos.ravel(), self.data.qvel.ravel()])

    def step(self, action):
        # 执行动作并前向模拟
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        # 在这里自定义 reward
        reward = -np.linalg.norm(obs)   # 仅示例

        # 定义终止条件
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        # 重置初始状态
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.01
        self.set_state(qpos, qvel)
        return self._get_obs()
