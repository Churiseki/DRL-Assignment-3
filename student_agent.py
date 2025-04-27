import gym

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

    def act(self, observation):
        return self.action_space.sample()

import gym
import numpy as np
import cv2
import torch
import os
from random import *
from gym import ObservationWrapper, Wrapper
from gym.spaces import Box



class MaxAndSkipFrameWrapper(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        return self.env.reset()


class FrameDownsampleWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)


class ImageToPyTorchWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = Box(
            low=0.0, high=1.0, shape=(old_shape[2], old_shape[0], old_shape[1]), dtype=np.float32
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


from collections import deque

class FrameBufferWrapper(Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(low=0.0, high=1.0, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_observation()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return np.concatenate(list(self.frames), axis=0)



class NormalizeFloats(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0



from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import make

def wrap_environment(env_name: str, action_space: list) -> Wrapper:
    env = make(env_name)
    env = JoypadSpace(env, action_space)
    env = MaxAndSkipFrameWrapper(env)
    env = FrameDownsampleWrapper(env)
    env = ImageToPyTorchWrapper(env)
    env = FrameBufferWrapper(env, 4)
    env = NormalizeFloats(env)
    return env

from collections import deque

# 建一個全域的 frame buffer
test_frame_buffer = deque(maxlen=4)

def preprocess_observation(obs):
    # 1. 轉成灰階
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    # 2. 縮小解析度
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    # 3. 增加 channel 維度
    obs = np.expand_dims(obs, axis=0)  # (1, 84, 84)
    # 4. 轉為 float 並 normalize
    obs = obs.astype(np.float32) / 255.0

    # 5. 把這張frame放進 buffer
    test_frame_buffer.append(obs)

    # 6. 如果 buffer 還沒滿，就補一樣的frame
    while len(test_frame_buffer) < 4:
        test_frame_buffer.append(obs)

    # 7. 把最近的4張frame疊起來 (4, 84, 84)
    stacked_obs = np.concatenate(list(test_frame_buffer), axis=0)
    
    return stacked_obs


import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from random import *
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# DQN define start

class DQNModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNModel, self).__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions
        
        # Define the feature extraction layers (convolutions)
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        x = self.features(x).view(x.size()[0], -1)
        return self.fc(x)
    
    def feature_size(self):
        # Pass a dummy tensor to get the size after feature extraction
        with torch.no_grad():
            return self.features(torch.zeros(1, *self._input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon, device):
        if np.random.rand() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            # action = randrange(self._num_actions)
            action = randrange(4) + 2
        return action
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]
    
    def __len__(self):
        return len(self.memory)
    
class DQNAgent():
    def __init__(self, model, target_model, replay_mem, optimizer, device,
                 batch_size=32, gamma=0.99, initial_learning=1000,
                 target_update_frequency=1000):
        self.model = model
        self.target_model = target_model
        self.replay_mem = replay_mem
        self.optimizer = optimizer
        self.device = device
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.initial_learning = initial_learning
        self.target_update_frequency = target_update_frequency

        self.epsilon = 1.0  # Initial epsilon for epsilon-greedy

    def act(self, state):
        # self.update_epsilon(episode_idx)
        action = self.model.act(state, self.epsilon, self.device)
        return action
    
    def process(self, episode_idx, state, action, reward, next_state, done):
        self.replay_mem.push(state, action, reward, next_state, done)
        self.train(episode_idx)
    
    def train(self, episode_idx):
        if len(self.replay_mem) > self.initial_learning:
            if episode_idx % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer.zero_grad()
            self.td_loss_backprop()
            self.optimizer.step()
    
    def td_loss_backprop(self):
        transitions = self.replay_mem.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state = Variable(torch.FloatTensor(np.float32(batch.state))).to(self.device)
        action = Variable(torch.LongTensor(batch.action)).to(self.device)
        reward = Variable(torch.FloatTensor(batch.reward)).to(self.device)
        next_state = Variable(torch.FloatTensor(np.float32(batch.next_state))).to(self.device)
        done = Variable(torch.FloatTensor(batch.done)).to(self.device)
        
        q_values = self.model(state)
        next_q_values = self.target_model(next_state)
        
        a_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        
        loss = (a_value - expected_q_value.detach()).pow(2)
        loss = loss.mean()
        loss.backward()

    def update_epsilon(self):
        # Epsilon decay strategy (ε-greedy)
        self.epsilon = max(0.001, self.epsilon * 0.9995)
    def save(self, filename='mario.pth'):
        """
        Save model state, target model state, optimizer state, and epsilon.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename='mario.pth'):
        """
        Load model state, target model state, optimizer state, and epsilon.
        """
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {filename}")
        else:
            print(f"Error: {filename} not found")
# DQN define end

# initialize the environment

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import time
import gym
import argparse
import torch
import torch.optim as optim
from collections import deque
from tqdm import tqdm

penalty_act = [6, 7, 8, 9]

env = wrap_environment("SuperMarioBros-v0", COMPLEX_MOVEMENT)
state = env.reset()
print(state.shape)  # 應該會是 (4, 84, 84)

input_shape = env.observation_space.shape
num_actions = env.action_space.n
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DQNModel(input_shape, num_actions).to(device)
target_model = DQNModel(input_shape, num_actions).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-4)


model.to(device)
target_model.to(device)

# 初始化 replay memory
replay_mem = ReplayMemory(20000)

# 創建 agent
agent = DQNAgent(model, target_model, replay_mem, optimizer, device,
                 batch_size=64, gamma=0.99,
                 initial_learning=10000,
                 target_update_frequency=1000)
agent.load()
agent.epsilon = 0.4

import gym

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)

    def act(self, observation):
        state = preprocess_observation(observation)
        action = agent.act(state)
        return action
        # return randrange(4) + 2