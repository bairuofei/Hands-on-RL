from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, info = env.reset(seed=0)
                done = False
                while not done:
                    action = agent.take_action(state)  # actor选择动作
                    # next_state, reward, done, _ = env.step(action)  # gym 0.18.3写法
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)  
                agent.update(transition_dict)  # 更新策略
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    # GAE算法计算优势函数
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()  # 时间越晚的advantage，权重修正越小
    return torch.tensor(advantage_list, dtype=torch.float)
                
                
                
import os
import cv2

class ManualRecordVideo:
    def __init__(self, env, video_folder="./videos", episode_trigger=None, fps=30):
        self.env = env
        self.video_folder = video_folder
        self.episode_trigger = episode_trigger if episode_trigger else lambda x: True
        self.fps = fps
        self.episode_count = 0
        self.frames = []
        self.recording = False
        
        os.makedirs(video_folder, exist_ok=True)
    
    def reset(self, **kwargs):
        # Check if we should record this episode
        self.recording = self.episode_trigger(self.episode_count)
        if self.recording:
            print(f"Recording episode {self.episode_count}")
        
        obs, info = self.env.reset(**kwargs)
        
        if self.recording:
            frame = self.env.render()
            self.frames = [frame]  # Start fresh
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.recording:
            frame = self.env.render()
            self.frames.append(frame)
        
        if terminated or truncated:
            if self.recording and len(self.frames) > 0:
                self._save_video()
            self.episode_count += 1
        
        return obs, reward, terminated, truncated, info
    
    def _save_video(self):
        if len(self.frames) == 0:
            return
            
        height, width, layers = self.frames[0].shape
        filename = f'{self.video_folder}/episode_{self.episode_count}.mp4'
        
        # Use XVID codec which is more compatible
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        
        for frame in self.frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved: {filename} ({len(self.frames)} frames)")
        self.frames = []
    
    def close(self):
        if self.recording and len(self.frames) > 0:
            self._save_video()
        self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)