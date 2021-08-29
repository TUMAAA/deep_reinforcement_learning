# This code is adopted from the original code provided by Udacity DRLND repo:
# https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

from .model import Actor, Critic

DEBUG = False
GAMMA = 0.99  # discount factor to compute discounted returns
TAU = 1e-3  # for soft update of target parameters
NOISE_VARIANCE = 1.0


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 device,
                 replay_buffer,
                 state_size,
                 action_size,
                 random_seed,
                 batch_size,
                 num_competing_agents,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 time_steps_before_training=20,
                 num_trainings_per_update=1,
                 noise_decay=1e-6,
                 num_episodes_to_increase_num_trainings=150,
                 weight_decay=0.0,
                 clip_grad_norm=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.weight_decay = weight_decay
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.num_competing_agents = num_competing_agents
        self.noise_decay = noise_decay
        self.time_steps_before_training = time_steps_before_training
        self.num_trainings_per_update = num_trainings_per_update
        self.num_episodes_to_increase_num_trainings = num_episodes_to_increase_num_trainings
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # not using list*num_competing_agents to
        # ensure each agent has a different random set of weights
        self.actors_local = [Actor(state_size, action_size, random_seed, fc1_units=64, fc2_units=64).to(
            self.device) for i_competing_agent in range(num_competing_agents)]

        self.actors_target = [Actor(state_size, action_size, random_seed, fc1_units=64, fc2_units=64).to(
            self.device) for i_competing_agent in range(num_competing_agents)]

        self.actor_optimizers = [optim.Adam(self.actors_local[i_competing_agent].parameters(),
                                            lr=self.lr_actor,
                                            weight_decay=self.weight_decay) for i_competing_agent in
                                 range(self.num_competing_agents)]

        joint_state_size = state_size * num_competing_agents
        joint_action_size = action_size * num_competing_agents
        self.critics_local = [
            Critic(joint_state_size, joint_action_size, random_seed,
                   fcs1_units=200, fc2_units=100).to(self.device) for i_competing_agent in
            range(self.num_competing_agents)]

        self.critics_target = [
            Critic(joint_state_size, joint_action_size, random_seed,
                   fcs1_units=200, fc2_units=100).to(self.device) for i_competing_agent in
            range(self.num_competing_agents)]

        self.critic_optimizers = [optim.Adam(self.critics_local[i_competing_agent].parameters(), lr=self.lr_critic,
                                             weight_decay=self.weight_decay) for i_competing_agent in
                                  range(self.num_competing_agents)]

        # Noise process
        self.noise_variance = NOISE_VARIANCE
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = replay_buffer

    def step(self, time_step, i_episode, agents_states, agents_actions, agents_rewards, agents_next_states,
             agents_dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(agents_states, agents_actions, agents_rewards, agents_next_states, agents_dones)

        if time_step % self.time_steps_before_training != 0:
            return

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            num_trainings = self.num_trainings_per_update * (
                    int(i_episode / self.num_episodes_to_increase_num_trainings) + 1)
            for i in range(num_trainings):
                joint_experiences_batch = self.memory.sample()
                self.learn(joint_experiences_batch, GAMMA)
                if DEBUG:
                    print(
                        f"\rTraining. i_episode: {i_episode}, time_step: {time_step}",
                        end="")
                    time.sleep(0.1)

    def act(self, state, i_agent, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state[i_agent]).float().to(self.device)
        self.actors_local[i_agent].eval()
        with torch.no_grad():
            action = self.actors_local[i_agent](state).cpu().data.numpy()
        self.actors_local[i_agent].train()
        if add_noise:
            self.noise_variance = self.noise_variance * self.noise_decay
            action += self.noise_variance * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise_variance = NOISE_VARIANCE
        self.noise.reset()

    def learn(self, joint_experiences_batch, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        agents_states_batch, agents_actions_batch, agents_rewards_batch, agents_next_states_batch, agents_dones_batch = joint_experiences_batch
        assert len(agents_states_batch) == self.num_competing_agents

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state agents_actions_batch and Q values from target models
        agents_next_actions_batch = [self.actors_target[i](agents_next_states_batch[i]) for i in
                                     range(self.num_competing_agents)]

        joint_next_states_batch = torch.cat(agents_next_states_batch, dim=1)
        joint_next_actions_batch = torch.cat(agents_next_actions_batch, dim=1)
        Q_targets_next = [self.critics_target[i](joint_next_states_batch, joint_next_actions_batch) for i in
                          range(self.num_competing_agents)]
        # Compute Q targets for current agents_states_batch (y_i)
        Q_targets = [agents_rewards_batch[i] + (gamma * Q_targets_next[i] * (1 - agents_dones_batch[i])) for i in
                     range(self.num_competing_agents)]

        # Compute critic loss
        joint_states_batch = torch.cat(agents_states_batch, dim=1)
        joint_actions_batch = torch.cat(agents_actions_batch, dim=1)
        Q_expected = [self.critics_local[i](joint_states_batch, joint_actions_batch) for i in
                      range(self.num_competing_agents)]
        critics_losses = [F.mse_loss(Q_expected[i], Q_targets[i]) for i in
                       range(self.num_competing_agents)]
        # Minimize the loss
        for i in range(self.num_competing_agents):
            self.critic_optimizers[i].zero_grad()
            critics_losses[i].backward(retain_graph=True)
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.critics_local[i].parameters(), 1)
            self.critic_optimizers[i].step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_preds = [self.actors_local[i](agents_states_batch[i]) for i in
                       range(self.num_competing_agents)]
        joint_actions_preds = torch.cat(actions_preds,dim=1)
        actors_losses = [-self.critics_local[i](joint_states_batch, joint_actions_preds).mean() for i in
                       range(self.num_competing_agents)]

        # Minimize the loss
        for i in range(self.num_competing_agents):
            self.actor_optimizers[i].zero_grad()
            actors_losses[i].backward(retain_graph=True)
            self.actor_optimizers[i].step()


        # ----------------------- update target networks ----------------------- #
        for i in range(self.num_competing_agents):
            self.soft_update(self.critics_local[i], self.critics_target[i], TAU)
            self.soft_update(self.actors_local[i], self.actors_target[i], TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


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
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.gauss(mu=0.0,sigma=0.1) for i in range(len(x))])
        self.state = x + dx
        return self.state
