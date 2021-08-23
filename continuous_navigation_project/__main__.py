from unityagents import UnityEnvironment
import numpy as np
import time
import matplotlib.pyplot as plt

# import wandb
# wandb.init(entity="local@wandb.com")

env = UnityEnvironment(file_name='Reacher_Twenty_Linux_NoVis/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

from collections import deque
from continuous_navigation_project.agent import Agent
import torch


def ddpg(agent, n_episodes=1000, max_t=300, print_every=100):
    mean_episode_score_deque = deque(maxlen=print_every)
    scores_global = []
    episode_durations = []
    for i_episode in range(1, n_episodes + 1):
        start_time = time.time()
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.rewards)
        states = env_info.vector_observations
        agent.reset()
        episode_score_per_agent = np.zeros(num_agents)
        for t in range(max_t):

            # action = agent.act(state)
            # next_state, reward, done, _ = env.step(action)
            # agent.step(state, action, reward, next_state, done)
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            for i_agent in range(num_agents):
                agent.step(states[i_agent].astype(np.float32),
                           t,
                           i_episode,
                           actions[i_agent],
                           rewards[i_agent],
                           next_states[i_agent].astype(np.float32),
                           dones[i_agent])

            states = next_states
            episode_score_per_agent += rewards
            if np.any(dones):
                break
        episode_durations.append(time.time() - start_time)
        mean_episode_score = np.mean(episode_score_per_agent)
        mean_episode_score_deque.append(mean_episode_score)
        scores_global.append(mean_episode_score)
        print('\rEpisode {}\tMean (over agents) episode score: {:.2f}. '
              'Duration: {:.1f}s'.format(i_episode,
                                         mean_episode_score_deque[-1],
                                         episode_durations[-1]), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print(
                '\rEpisode {}\tAverage mean agent Score: {:.2f}. Average duration {:.1f}s. Averages over last {} episodes.'.format(
                    i_episode, np.mean(mean_episode_score_deque), np.mean(episode_durations[-print_every:]),
                    print_every))
        if np.mean(scores_global[-100:]) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage mean score: {:.2f}'.format(i_episode - 100,
                                                                                              np.mean(scores_global[
                                                                                                      -100:])))
            torch.save(agent.qnetwork_local.state_dict(),
                       "trained_models/" + "checkpoint_dropout_p{}_hiddenlayers{}.pth".format(
                           agent.qnetwork_local.drop_p, agent.qnetwork_local.hidden_layers_config))
            break

    return scores_global, episode_durations


MAX_T = 400
agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)
scores_global, episode_durations = ddpg(agent=agent, n_episodes=250, max_t=MAX_T, print_every=20)

fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(np.arange(1, len(scores_global) + 1), scores_global)
plt.ylabel('Accum Rewards (Score)')
plt.xlabel('Episode #')
ax = fig.add_subplot(212)
plt.plot(np.arange(1, len(episode_durations) + 1), episode_durations)
plt.ylabel('Training Duration [s]')
plt.xlabel('Episode #')
plt.show()
