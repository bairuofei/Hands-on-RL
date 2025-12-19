from typing import Dict, Tuple, Union

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class Galaxea_r1Pro(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "/home/ruofei/code/learning/Hands-on-RL/examples/galaxea_r1/galaxea_r1pro_simple/r1_pro.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 50,  # previous 1.25
        ctrl_cost_weight: float = 0.05,
        contact_cost_weight: float = 5e-7,
        contact_cost_range: Tuple[float, float] = (-np.inf, 10.0),
        healthy_reward: float = 5.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.8, 1.5),
        reset_noise_scale: float = 1e-2,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(  # 用于保存所有环境参数
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            contact_cost_range,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = 3 + self.data.qpos.size + self.data.qvel.size + self.data.xipos.size - 3
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "target_xyz": 3,
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
            "body_pos": self.data.xipos.size - 3,
            "cinert": self.data.cinert[1:].size * include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * include_cvel_in_observation,
            "qfrc_actuator": self.data.qvel.size * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "ten_length": 0,
            "ten_velocity": 0,
        }
        
        self.target_range = 2.0
        self.rng = np.random.default_rng(seed = 12) 
        # generate new target xyz
        self.target_xyz = self.rng.uniform(low=-self.target_range, high=self.target_range, size=3)
        self.target_xyz[2] = 1.2  # 保持在地面
        self.target_tolerance = 0.2  # 目标点容忍范围

    @property
    def is_healthy(self):
        """torso_link4的z坐标在[0.8, 1.5]之间
        初始值如下：当前机器人保持直立
        body[ 8]: torso_link1, pos=(-0.078, 0.003, 0.574)
        body[ 9]: torso_link2, pos=(-0.074, 0.014, 0.900)
        body[10]: torso_link3, pos=(-0.079, -0.005, 1.063)
        body[11]: torso_link4, pos=(-0.078, -0.000, 1.437)
        """
        min_z, max_z = self._healthy_z_range
        return min_z < self.data.xipos[11][2] < max_z

    def _get_obs(self):
        target_xyz = self.target_xyz - self.data.xipos[25]
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()
        body_pos = self.data.xipos[1:].flatten()  # 去掉第一个，是全局位置

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flatten()
        else:
            com_inertia = np.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flatten()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator.flatten()
        else:
            actuator_forces = np.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:].flatten()
        else:
            external_contact_forces = np.array([])
            
        return np.concatenate(
            (   
                target_xyz,
                position,
                velocity,
                body_pos,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )
        
    def _get_reset_info(self):
        return {
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
        
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        
        # generate new target xyz
        self.target_xyz = self.rng.uniform(low=-self.target_range, high=self.target_range, size=3)
        self.target_xyz[2] = 1.2  # 保持在地面
        
        observation = self._get_obs()
        return observation

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        
        x_pos_before = self.data.qpos[0]
        
        # 调用mujoco仿真
        self.do_simulation(action, self.frame_skip)
        
        x_pos_after = self.data.qpos[0]
        
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity
        
        x_pos_velocity = (x_pos_after - x_pos_before) / self.dt

        
        
        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_pos_velocity, x_velocity, action)
        dist = np.linalg.norm(self.data.xipos[25] - self.target_xyz, ord=2)
        terminated = False
        if ((not self.is_healthy) and self._terminate_when_unhealthy) or dist > 5.0:
            terminated = True
        # terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def get_healthy_reward(self):
        return self.is_healthy * self._healthy_reward
    
    def get_contact_cost(self):
        contact_forces = self.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost
    
    def get_target_reward(self):
        dist = np.linalg.norm(self.data.xipos[25] - self.target_xyz, ord=2)
        target_reward = (np.exp(-dist) - 1) * 10
        if dist < self.target_tolerance:
            target_reward = 15.0  
        return target_reward
    
    def _get_rew(self, x_pos_velocity: float, x_velocity: float, action):
        healthy_reward = self.get_healthy_reward()   # reward: [0, 5]
        forward_reward = np.clip(self._forward_reward_weight * x_pos_velocity, -10, 10)
        forward_reward = 0
        
        control_cost = np.clip(self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl)), 0, 10)  # reward: (-inf, 0]
        contact_cost = self.get_contact_cost()     # (-np.inf, 10.0)
        
        target_reward = self.get_target_reward()  # [-10, 15]

        reward = forward_reward + healthy_reward +target_reward - control_cost - contact_cost

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -control_cost,
            "reward_contact": -contact_cost,
            "reward_target": target_reward,
        }

        return reward, reward_info




