import os
import time
from collections import deque

import numpy as np
import torch

# This module should not have any configs of the agent
from collaboration_and_competition_project import brain_name, env

EPISODE_LENGTH_FOR_AVERAGING = 100  # num episodes over which score avg should be > MIN_AVG_SCORE_OVER_LAST_HUNDRED_EPISODES_TO_BEAT
MIN_AVG_SCORE_OVER_LAST_HUNDRED_EPISODES_TO_BEAT = 0.5
TRAINED_MODELS_DIR = "trained_models"


def save_checkpoints(agent):
    """
    saves a checkpoint of the models of the agent
    :param agent: continuous navigation agent that has a local and target actor and a local and target critic
    """
    for i in range(len(agent.actors_local)):
        torch.save(agent.actors_local[i].state_dict(), os.path.join(TRAINED_MODELS_DIR, 'checkpoint_actor_{}.pth'.format(i)))
    for i in range(len(agent.critics_local)):
        torch.save(agent.critics_local[i].state_dict(), os.path.join(TRAINED_MODELS_DIR, 'checkpoint_critic_{}.pth'.format(i)))


def run_maddpg(agent,
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
    max_episode_score_deque = deque(maxlen=print_every)
    scores_global = []
    episode_durations = []
    episode_timestep_reached=[-1.0]*n_episodes
    global_start_time = time.time()
    for i_episode in range(1, n_episodes + 1):
        episode_start_time = time.time()
        if i_episode % episodes_to_make_target_equal_to_local == 0:
            reset_target_to_local(agent)
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.rewards)
        states = env_info.vector_observations
        agent.reset()
        episode_score_per_agent = np.zeros(num_agents)
        for t in range(max_t):
            # if t % 100 == 0:
            #     print("\rreached noise variance: {} at t {}".format(agent.noise_variance,t),end="")
            #     time.sleep(0.1)
            actions = [agent.act(states, i_agent) for i_agent in range(num_agents)]
            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished
            agent.step(t,
                       i_episode,
                       states.astype(np.float32),
                       actions,
                       rewards,
                       next_states.astype(np.float32),
                       dones)

            states = next_states
            episode_score_per_agent += rewards
            if np.any(dones):
                episode_timestep_reached[i_episode-1]=t
                break

        episode_durations.append(time.time() - episode_start_time)
        max_episode_score = np.max(episode_score_per_agent)
        max_episode_score_deque.append(max_episode_score)
        scores_global.append(max_episode_score)
        print('\rEpisode {}\tMax (over agents) episode score: {:.2f}. '
              'Duration: {:.1f}s'.format(i_episode,
                                         max_episode_score_deque[-1],
                                         episode_durations[i_episode-1]), end="")
        save_checkpoints(agent)
        if i_episode % print_every == 0:
            print(
                '\rEpisode {}\tAverage max agent Score: {:.2f}. Average duration {:.1f}s. Avg timestep reached {}'.format(
                    i_episode, np.mean(max_episode_score_deque), np.mean(episode_durations[-print_every:]),
                    np.mean(episode_timestep_reached[i_episode-print_every:i_episode])))

        if np.mean(scores_global[-EPISODE_LENGTH_FOR_AVERAGING:]) >= MIN_AVG_SCORE_OVER_LAST_HUNDRED_EPISODES_TO_BEAT:
            print('\nEnvironment solved in {:d} episodes!\tAverage mean score over last 100 episodes: {:.2f}'
                  .format(i_episode - EPISODE_LENGTH_FOR_AVERAGING,
                          np.mean(scores_global[i_episode-EPISODE_LENGTH_FOR_AVERAGING:i_episode])))
            save_checkpoints(agent)
            break
    print("")
    print("DONE ----------------------")
    print("Total time consumed: {:.1f}m".format((time.time() - global_start_time) / 60.0))
    episode_timestep_reached_array = np.asarray(episode_timestep_reached)
    episode_timestep_reached_array[episode_timestep_reached_array < 0 ] = max_t
    return scores_global, episode_durations, episode_timestep_reached


def reset_target_to_local(agent):
    print("\rresetting target to be equal to local", end="")
    time.sleep(0.5)
    for i in range(len(agent.actors_local)):
        agent.soft_update(agent.actors_local[i], agent.actors_target[i], tau=1.0)
    for i in range(len(agent.critics_local)):
        agent.soft_update(agent.critics_local[i], agent.critics_target[i], tau=1.0)
