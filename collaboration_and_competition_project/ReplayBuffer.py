from collections import deque, namedtuple
import random

import torch
from pandas import np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, random_seed, num_collaborating_agents,
                 device: torch.device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.num_collaborating_agents = num_collaborating_agents
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.agent_experience = namedtuple("Experience",
                                           field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(random_seed)

    def add(self, state: list, action: list, reward: list, next_state: list, done: list):
        """Add a new joint_experience to memory.
        An joint_experience contains the list of agent experiences collected in the same time step
        """

        num_agents = len(state)
        joint_experience = [self.agent_experience(state[i], action[i], reward[i], next_state[i], done[i]) for i in
                            range(num_agents)]
        # e = self.joint_experience(state, action, reward, next_state, done)
        self.memory.append(joint_experience)

    def sample(self):
        """Randomly sample a batch of joint_experiences from memory."""
        # TODO: document
        joint_experiences = random.sample(self.memory, k=self.batch_size)

        agents_states_batch = [
            torch.from_numpy(np.vstack([e[i].state for e in joint_experiences if e is not None])).float().to(
                self.device) for i in range(self.num_collaborating_agents)]
        agents_actions_batch = [
            torch.from_numpy(np.vstack([e[i].action for e in joint_experiences if e is not None])).float().to(
                self.device) for i in range(self.num_collaborating_agents)]
        agents_rewards_batch = [
            torch.from_numpy(np.vstack([e[i].reward for e in joint_experiences if e is not None])).float().to(
                self.device) for i in range(self.num_collaborating_agents)]
        agents_next_states_batch = [
            torch.from_numpy(np.vstack([e[i].next_state for e in joint_experiences if e is not None])).float().to(
                self.device) for i in range(self.num_collaborating_agents)]
        agents_dones_batch = [torch.from_numpy(
            np.vstack([e[i].done for e in joint_experiences if e is not None]).astype(np.uint8)).float().to(
            self.device) for i in range(self.num_collaborating_agents)]

        return (
            agents_states_batch, agents_actions_batch, agents_rewards_batch, agents_next_states_batch,
            agents_dones_batch)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
