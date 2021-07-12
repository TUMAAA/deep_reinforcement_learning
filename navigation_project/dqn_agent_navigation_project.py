import numpy as np
import random
from collections import namedtuple, deque

from model.model import QNetwork

import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"found device: {device}")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, drop_p=0.0, hidden_layers_config=[64, 64],
                 duelling_networks=False,
                 priority_experience=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, drop_p=drop_p,
                                       hidden_layers_config=hidden_layers_config).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, drop_p=drop_p,
                                        hidden_layers_config=hidden_layers_config).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, priority_experience=priority_experience)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # ANAS
        self.running_loss = []
        print(f"Device to be used: {device}")
        self.criterion = torch.nn.MSELoss(size_average=True, reduce=True) # the course jupyter env has old ver of pytorch https://pytorch.org/docs/0.4.0/nn.html#mseloss

        self.duelling_networks = duelling_networks

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        "*** YOUR CODE HERE ***"

        # ANAS CODE
        self.qnetwork_local.train()
        
        output = self.qnetwork_local(states)
        output_for_chosen_actions = output.gather(dim=1, index=actions)

        # You can replace max with mean just like SARSAmean
        expected_output = rewards + gamma * self.compute_max_term(next_states).detach() * (1 - dones)

        loss = self.criterion(output_for_chosen_actions, expected_output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def compute_max_term(self, next_states):
        if self.duelling_networks:
            actions_maximizing_q = self.qnetwork_local(next_states).max(dim=1, keepdim=True)[1]
            return self.qnetwork_target(next_states).gather(dim=1, index=actions_maximizing_q)
        else:
            return self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0]

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, priority_experience=False):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

        self.priority_experience = priority_experience
        self.a = 0.8

    def add(self, state, action, reward, next_state, done, probability=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append((e, probability))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if self.priority_experience:
            experiences = self.get_experiences_with_priority()
        else:
            experiences = self.get_experiences()

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def get_experiences(self):
        return [memory[0] for memory in random.sample(self.memory, k=self.batch_size)]

    def get_experiences_with_priority(self):
        weights = [memory[1] ** self.a for memory in self.memory]
        assert None not in weights
        return [memory[0] for memory in random.choices(self.memory, k=self.batch_size, weights=weights)]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
