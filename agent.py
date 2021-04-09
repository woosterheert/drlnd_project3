import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from utils import device
import copy
import time


class Agent:

    def __init__(self, online_value_network, target_value_network, online_policy_network, target_policy_network,
                 world, memory, config, train_mode=True):
        self.online_value_network = online_value_network.to(device)
        self.target_value_network = target_value_network.to(device)
        self.online_policy_network = online_policy_network.to(device)
        self.target_policy_network = target_policy_network.to(device)
        self.world = world
        self.train_mode = train_mode
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = memory
        self.nr_steps = 0
        self.total_reward = 0
        self.noise_scale = 1
        self.finished = False
        self.all_scores = []
        self.update_freq = config['update_freq']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.max_nr_steps = config['max_nr_steps']
        self.output_freq = config['output_freq']
        self.nr_episodes = config['nr_episodes']
        self.actions_boundaries = config['action_boundaries']
        self.noise_scale = config['noise_scale_init']
        self.noise_scale_min = config['noise_scale_min']
        self.noise_decl = config['noise_decl']
        self.n_agents = config['n_agents']
        self.nr_updates_per_step = config['nr_updates']
        self.value_optimizer = optim.Adam(self.online_value_network.parameters(), lr=config['lr'])
        self.policy_optimizer = optim.Adam(self.online_policy_network.parameters(), lr=config['lr'])
        self.state = None
        self.noise = OUNoise(self.world.action_space_size, 0)

    def initialize_world(self, noise_scale):
        if self.world.type == 'gym':
            self.state = self.world.env.reset()
        elif self.world.type == 'unity_vector':
            env_info = self.world.env.reset(train_mode=self.train_mode)[self.world.brain_name]
            self.state = env_info.vector_observations.reshape(1, -1)
        elif self.world.type == 'unity_visual':
            env_info = self.world.env.reset(train_mode=self.train_mode)[self.world.brain_name]
            self.state = env_info.visual_observations[0]
        elif self.world.type == 'unity_online':
            env_info = self.world.env.reset(train_mode=self.train_mode)[self.world.brain_name]
            self.state = env_info.vector_observations[0]
        self.nr_steps = 0
        self.total_reward = np.zeros(self.n_agents)
        self.finished = False
        self.noise_scale = noise_scale

    def select_action(self):
        # run inference on the local network
        self.online_policy_network.eval()
        with torch.no_grad():
            greedy_action = self.online_policy_network(torch.from_numpy(self.state).float().to(device))
            greedy_action = greedy_action.cpu().data.numpy()
        self.online_policy_network.train()

        noisy_action = greedy_action + self.sample_noise()
        # noisy_action = greedy_action + self.noise_scale * self.noise.sample()
        actions = np.clip(noisy_action, self.actions_boundaries[0], self.actions_boundaries[1])
        return actions

    def sample_noise(self):
        return np.random.normal(loc=0., scale=self.noise_scale, size=(1, self.world.action_space_size))

    def take_action(self, action):
        if self.world.type == 'gym':
            next_state, reward, done, _ = self.world.env.step(action)
        elif self.world.type == 'unity_vector':
            env_info = self.world.env.step(action.reshape(self.n_agents, -1))[self.world.brain_name]
            next_state = env_info.vector_observations.reshape(1, -1)
            reward = env_info.rewards
            done = env_info.local_done
        elif self.world.type == 'unity_visual':
            env_info = self.world.env.step(action)[self.world.brain_name]
            next_state = env_info.visual_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
        elif self.world.type == 'unity_online':
            env_info = self.world.env.step(action)[self.world.brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
        self.nr_steps += 1
        return next_state, reward, done

    def process_results(self, actions, results):
        next_states, rewards, dones = results
        for s, a, r, ns, d in zip(self.state, actions, rewards, next_states, dones):
            self.memory.add(self.experience(s, a, r, ns, d))

        if self.nr_steps % self.update_freq == 0:
            if self.memory.enough_data:
                for i in range(self.nr_updates_per_step):
                    # print(f'update {i}')
                    self.update()
                self.noise_scale = max(self.noise_scale_min, self.noise_scale * self.noise_decl)

        self.total_reward += np.array(rewards)
        self.state = next_states
        if np.any(dones) or self.nr_steps > self.max_nr_steps:
            self.finished = True

    def update(self):
        # get a batch
        states, actions, rewards, next_states, dones = self.memory.sample()

        # update the value network
        selected_actions_next_states = self.target_policy_network(next_states)
        q_values_ans = self.target_value_network(next_states, selected_actions_next_states)
        target_q = rewards + self.gamma * q_values_ans * (1 - dones.unsqueeze(1))
        estimated_q = self.online_value_network(states, actions)

        value_loss = F.mse_loss(estimated_q, target_q)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm(self.online_value_network.parameters(), 1)
        self.value_optimizer.step()

        # update the policy network
        predicted_actions = self.online_policy_network(states)
        corresponding_q_values = self.online_value_network(states, predicted_actions)

        policy_loss = -corresponding_q_values.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm(self.online_policy_network.parameters(), 1)
        self.policy_optimizer.step()

        # update the target network
        self.update_target(self.target_value_network, self.online_value_network)
        self.update_target(self.target_policy_network, self.online_policy_network)

    def update_target(self, target_network, local_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def train(self):
        scores = deque(maxlen=self.output_freq)
        current_time = time.time()
        for i in range(1, self.nr_episodes + 1):
            self.initialize_world(self.noise_scale)
            while not self.finished:
                a = self.select_action()
                res = self.take_action(a)
                self.process_results(a, res)
            scores.append(self.total_reward)
            self.all_scores.append(self.total_reward)
            if i % self.output_freq == 0:
                np_scores = np.vstack(scores)
                max_scores = np.max(np_scores, axis=1)
                score_to_print = round(np.mean(max_scores), 2)
                print(f'finished episodes {i-self.output_freq+1} - {i} in {time.time() - current_time} s. with avg reward {score_to_print}')
                current_time = time.time()

        return self.all_scores

    def run(self):
        total_reward = np.zeros(self.n_agents)
        self.initialize_world(0)
        while not self.finished:
            a = self.select_action()
            next_state, reward, done = self.take_action(a)
            total_reward += np.array(reward)
            self.state = next_state
            if np.any(done) or self.nr_steps > self.max_nr_steps:
                self.finished = True
                print(f'Players got a maximum of {np.mean(total_reward)} points!')


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state