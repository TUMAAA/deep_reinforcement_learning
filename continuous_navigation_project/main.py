from unityagents import UnityEnvironment
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import torch
import os

from continuous_navigation_project.agent import Agent

MIN_AVG_SCORE_OVER_LAST_HUNDRED_EPISODES_TO_BEAT = 30.0
TRAINED_MODELS_DIR = "trained_models"

env = UnityEnvironment(file_name='Reacher_Twenty_Linux_NoVis/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


def generate_plot_name(attributes: dict):
    """
    Generates a title for the plots showing the learning curve. The title encodes the used configuration of the agent.
    :param attributes: dict of key-values that encode configuration var and its value (e.g. {"max_t":300}).
        Simply put everything you want to have in the plot title
    :return: string title
    """
    title = ""
    for key in attributes.keys():
        title += "{} {}\n".format(key, attributes[key])

    return title


def generate_training_plots(scores_global, episode_durations, attributes):
    """
    Generates subplots showing the mean score and training duration per episode
    :param scores_global: List of mean scores where each element represents an episode
    :param episode_durations: List of episode duration where each element represents an episode
    :param attributes: A dict where each key-value pair gets converted to a string added to the plot title
    """
    fig = plt.figure()
    ax = fig.add_subplot(413)
    plt.plot(np.arange(1, len(scores_global) + 1), scores_global)
    plt.ylabel('Accum Rewards (Score)')
    plt.xlabel('Episode #')
    max_y = np.max(scores_global)
    max_y = (int(max_y / 10) + 1) * 10
    plt.ylim(0, max_y)
    grid_step = 10
    ax.set_yticks(range(10, max_y, grid_step), minor=False)
    ax.yaxis.grid(True, which="major")

    ax = fig.add_subplot(414)
    num_episodes = len(episode_durations)
    plt.plot(np.arange(1, num_episodes + 1), episode_durations)
    plt.ylabel('Training Duration [s]')
    plt.xlabel('Episode #')
    title = generate_plot_name(attributes)
    fig.suptitle(title, fontsize=7)
    plt.show()


def ddpg(agent,
         n_episodes=1000,
         max_t=1000,
         print_every=100,
         episodes_to_make_target_equal_to_local=10):
    """
    Runs the DDPG algorithm on the given agent to solve the challenge
    :param agent: A DDPG actor-critic agent with local and target models for actor and critic
    :param n_episodes: Max number of episodes to run to try to solve the challenge
    :param max_t: Maximum number of timesteps per episode (the higher the better)
    :param print_every: The number of episodes for which a checkpoint is saved and the mean scores are averaged
    :param episodes_to_make_target_equal_to_local: The target models are set to the local models every time the episde
            index is a multiple of this (set very high if you don't want that and want to rely on soft update only).
    :return: A tuple of a list of mean agent score per episode and a list of episode duration
    """
    mean_episode_score_deque = deque(maxlen=print_every)
    scores_global = []
    episode_durations = []
    global_start_time = time.time()
    for i_episode in range(1, n_episodes + 1):
        episode_start_time = time.time()
        if i_episode % episodes_to_make_target_equal_to_local == 0:
            print("\rresetting target to be equal to local", end="")
            time.sleep(0.5)
            agent.soft_update(agent.actor_local, agent.actor_target, tau=1.0)
            agent.soft_update(agent.critic_local, agent.critic_target, tau=1.0)
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.rewards)
        states = env_info.vector_observations
        agent.reset()
        episode_score_per_agent = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            for i_agent in range(num_agents):
                agent.step(t,
                           i_episode,
                           i_agent,
                           states[i_agent].astype(np.float32),
                           actions[i_agent],
                           rewards[i_agent],
                           next_states[i_agent].astype(np.float32),
                           dones[i_agent])

            states = next_states
            episode_score_per_agent += rewards
            if np.any(dones):
                break
        episode_durations.append(time.time() - episode_start_time)
        mean_episode_score = np.mean(episode_score_per_agent)
        mean_episode_score_deque.append(mean_episode_score)
        scores_global.append(mean_episode_score)
        print('\rEpisode {}\tMean (over agents) episode score: {:.2f}. '
              'Duration: {:.1f}s'.format(i_episode,
                                         mean_episode_score_deque[-1],
                                         episode_durations[-1]), end="")
        save_checkpoints(agent)
        if i_episode % print_every == 0:
            print(
                '\rEpisode {}\tAverage mean agent Score: {:.2f}. Average duration {:.1f}s. Averages over last {} episodes.'.format(
                    i_episode, np.mean(mean_episode_score_deque), np.mean(episode_durations[-print_every:]),
                    print_every))
        if np.mean(scores_global[-100:]) >= MIN_AVG_SCORE_OVER_LAST_HUNDRED_EPISODES_TO_BEAT:
            print('\nEnvironment solved in {:d} episodes!\tAverage mean score over last 100 episodes: {:.2f}'
                  .format(i_episode - 100, np.mean(scores_global[-100:])))
            save_checkpoints(agent)
            break
    print("")
    print("DONE ----------------------")
    print("Total time consumed: {:.1f}m".format((time.time() - global_start_time) / 60.0))

    return scores_global, episode_durations


def save_checkpoints(agent):
    """
    saves a checkpoint of the models of the agent
    :param agent: continuous navigation agent that has a local and target actor and a local and target critic
    """
    torch.save(agent.actor_local.state_dict(), os.path.join(TRAINED_MODELS_DIR, 'checkpoint_actor.pth'))
    torch.save(agent.critic_local.state_dict(), os.path.join(TRAINED_MODELS_DIR, 'checkpoint_critic.pth'))


agent = Agent(state_size=state_size, action_size=action_size, random_seed=2, num_parallel_agents=num_agents,
              num_trainings_per_update=20,
              time_steps_before_training=20,
              batch_size=128,
              num_episodes_to_increase_num_trainings=180,
              lr_actor=1e-4,
              lr_critic=1e-4,
              make_local_target_weights_equal_at_init=True,
              clip_grad_norm=False)

episodes_to_make_target_equal_to_local = 5
max_timesteps_per_episode = 1000
scores_global, episode_durations = ddpg(agent=agent,
                                        n_episodes=130,
                                        max_t=max_timesteps_per_episode,
                                        print_every=20,
                                        episodes_to_make_target_equal_to_local=episodes_to_make_target_equal_to_local)
generate_training_plots(scores_global, episode_durations,
                        {"critic": agent.critic_local.__repr__(),
                         "actor": agent.actor_local.__repr__(),
                         "critic_optim": agent.critic_optimizer.__repr__().replace("\n", ", "),
                         "actor_optim": agent.actor_optimizer.__repr__().replace("\n", ", "),
                         "clip_grad_norm": agent.clip_grad_norm,
                         "batch_size": agent.batch_size,
                         "max_t": episodes_to_make_target_equal_to_local,
                         "time_steps_before_training": agent.time_steps_before_training,
                         "num_trainings_per_update": agent.num_trainings_per_update,
                         "num_episodes_to_increase_num_trainings": agent.num_episodes_to_increase_num_trainings,
                         "noise_decay": agent.noise_decay,
                         "episodes_to_make_target_equal_to_local": episodes_to_make_target_equal_to_local
                         })
