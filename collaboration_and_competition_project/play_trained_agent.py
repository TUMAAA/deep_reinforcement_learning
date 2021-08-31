import torch

from collaboration_and_competition_project import state_size, action_size, num_agents, env, brain_name, device
from collaboration_and_competition_project.agent import Agent
import numpy as np

from collaboration_and_competition_project.model import Actor


def act(state, actor_models, add_noise=True):
    """Returns actions for given state as per current policy."""
    num_agents = len(state)
    actions = [None]*num_agents
    for i_agent in range(num_agents):
        agent_specific_state = torch.from_numpy(state[i_agent]).float().to(device)
        actor_models[i_agent].eval()

        with torch.no_grad():
            actions[i_agent] = actor_models[i_agent](agent_specific_state).cpu().data.numpy()
        if add_noise:
            actions[i_agent] += 0.35*np.random.standard_normal(size=len(actions[i_agent]))
            actions[i_agent] = np.clip(actions[i_agent],-1,1)
    print("actions: {}. actions norms {}".format(actions, np.linalg.norm(actions,axis=0)))
    return actions


actors = [Actor(state_size=state_size,action_size=action_size,seed=2,fc1_units=64, fc2_units=64).to(device)]*num_agents
actors[0].load_state_dict(torch.load("trained_models/checkpoint_actor_0.pth"))
actors[1].load_state_dict(torch.load("trained_models/checkpoint_actor_1.pth"))

env_info = env.reset(train_mode=False)[brain_name]            # reset the environment
states = env_info.vector_observations                         # get the current state (for each agent)
scores = np.zeros(num_agents)                                 # initialize the score (for each agent)
for t in range(1000):
    actions = act(states,actors )                 # get actions from model (for each agent)
    env_info = env.step(actions)[brain_name]                  # send all actions to tne environment
    next_states = env_info.vector_observations                # get next state (for each agent)
    rewards = env_info.rewards                                # get reward (for each agent)
    dones = env_info.local_done                               # see if episode finished
    scores += env_info.rewards                                # update the score (for each agent)
    states = next_states                                      # roll over states to next time step
    if np.any(dones):                                         # exit loop if episode finished
        break

print('Total score (averaged over agents) this episode: {}'.format(np.max(scores)))