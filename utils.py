import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import random
import torch
from collections import deque
from unityagents import UnityEnvironment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Environment:
    """
    wrapper class that makes the necessary environment variables available
    in the same format for Gym as well as both the vector and visual
    environments from Unity
    """

    def __init__(self, env_type, env_name, n_agents):
        self.type = env_type
        self.name = env_name
        self.env = None
        self.observation_space_size = None
        self.action_space_size = None
        self.brain = None
        self.brain_name = None
        self.n_agents = n_agents
        self.initialize()

    def initialize(self):
        # gym implementation currently only supports 1D environments
        if self.type == 'gym':
            self.env = gym.make(self.name)
            self.env.seed(0)
            self.observation_space_size = self.env.observation_space.shape
            self.action_space_size = self.env.action_space.n
            self.observation_space_size = self.observation_space_size[0]

        elif self.type == 'unity_vector':
            self.env = UnityEnvironment(file_name=self.name)
            self.brain_name = self.env.brain_names[0]
            self.brain = self.env.brains[self.brain_name]
            self.action_space_size = self.brain.vector_action_space_size * self.n_agents
            self.observation_space_size = self.brain.vector_observation_space_size * \
                                          self.brain.num_stacked_vector_observations * self.n_agents

        elif self.type == 'unity_visual':
            self.env = UnityEnvironment(file_name=self.name)
            self.brain_name = self.env.brain_names[0]
            self.brain = self.env.brains[self.brain_name]
            self.action_space_size = self.brain.vector_action_space_size
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            self.observation_space_size = env_info.visual_observations[0].shape

        elif self.type == 'unity_online':
            self.env = self.name
            self.brain_name = self.env.brain_names[0]
            self.brain = self.env.brains[self.brain_name]
            self.action_space_size = self.brain.vector_action_space_size
            self.observation_space_size = self.brain.vector_observation_space_size

    def close(self):
        self.env.close()


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)
        self.enough_data = False

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.batch_size:
            self.enough_data = True

    def sample(self):
        sampled_experiences = random.sample(self.buffer, self.batch_size)

        states_pt = torch.from_numpy(np.vstack([e.state for e in sampled_experiences])).float().to(device)
        actions_pt = torch.from_numpy(np.vstack([e.action for e in sampled_experiences])).float()\
            .to(device)
        rewards_pt = torch.from_numpy(np.array([e.reward for e in sampled_experiences]).reshape(-1, 1)).float()\
            .to(device)
        next_states_pt = torch.from_numpy(np.vstack([e.next_state for e in sampled_experiences])).float().to(device)
        dones_pt = torch.from_numpy(np.array([e.done for e in sampled_experiences]).astype('int')).float().to(device)

        return states_pt, actions_pt, rewards_pt, next_states_pt, dones_pt


def plot_results(results, window_size, target_score):
    df = pd.DataFrame(results)
    ax = df.plot()
    df.rolling(window_size).mean().plot(ax=ax)
    plt.hlines(target_score, 0, len(df), 'g', label='target score')
    ax.legend(["score", "score rolling average"])
    plt.xlabel('episode')
