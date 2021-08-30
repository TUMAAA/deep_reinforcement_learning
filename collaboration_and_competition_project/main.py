from collaboration_and_competition_project.ReplayBuffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size # TODO: check value

import matplotlib.pyplot as plt
import numpy as np

from collaboration_and_competition_project import state_size, action_size, num_agents, device
from collaboration_and_competition_project.agent import Agent
from collaboration_and_competition_project.maddpg import run_maddpg


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


def generate_training_plots(scores_global, episode_durations, episode_timestep_reached, attributes):
    """
    Generates subplots showing the mean score and training duration per episode
    :param scores_global: List of mean scores where each element represents an episode
    :param episode_durations: List of episode duration where each element represents an episode
    :param attributes: A dict where each key-value pair gets converted to a string added to the plot title
    """
    fig = plt.figure(figsize=[8, 12])
    ax_score = fig.add_subplot(412)
    plt.plot(np.arange(1, len(scores_global) + 1), scores_global)
    plt.ylabel('Accum Rewards (Score)')
    plt.xlabel('Episode #')
    max_y = np.max(scores_global)
    max_y = (int(max_y / 1) + 1) * 1
    plt.ylim(0, max_y)
    # grid_step = 0.2
    # ax_score.set_yticks(np.linspace(1, max_y, grid_step), minor=False)
    ax_score.yaxis.grid(True, which="major")

    ax_ts_reached = fig.add_subplot(413)
    num_episodes = len(episode_timestep_reached)
    plt.plot(np.arange(1, num_episodes + 1), episode_timestep_reached)
    plt.ylabel('Timestep reached')
    plt.xlabel('Episode #')

    ax_duration = fig.add_subplot(414)
    num_episodes = len(episode_durations)
    plt.plot(np.arange(1, num_episodes + 1), episode_durations)
    plt.ylabel('Training Duration [s]')
    plt.xlabel('Episode #')

    title = generate_plot_name(attributes)
    fig.suptitle(title, fontsize=8)
    plt.show()


# episodes_to_make_target_equal_to_local = 200
# max_timesteps_per_episode = 6500

from collections import namedtuple

TrainingConfig = namedtuple(typename="TrainingConfig", field_names=["max_t",
                                                                    "batch_size",
                                                                    "num_trainings_per_update",
                                                                    "time_steps_before_training",
                                                                    "clip_grad_norm",
                                                                    "lr_actor",
                                                                    "lr_critic",
                                                                    "episodes_to_make_target_equal_to_local"])

configs = [
    # Compare effect of LR
    TrainingConfig(1000, 128, 1, 1, False, 1e-5, 1e-5, 500),
    TrainingConfig(1000, 128, 1, 1, False, 1e-4, 1e-4, 500),
    TrainingConfig(1000, 128, 1, 1, False, 1e-3, 1e-3, 500),

    # Compare effect batch size
    TrainingConfig(1000, 64, 1, 1, False, 1e-4, 1e-4, 500),

    # Compare clip grad norm
    TrainingConfig(1000, 128, 1, 1, True, 1e-4, 1e-4, 500),

    # Compare num time steps before training
    TrainingConfig(1000, 128, 1, 3, False, 1e-4, 1e-4, 500),
]
for config in configs:
    replay_buffer = ReplayBuffer(action_size=action_size,
                                 buffer_size=BUFFER_SIZE,
                                 batch_size=config.batch_size,
                                 random_seed=2,
                                 num_collaborating_agents=num_agents,
                                 device=device)

    agent = Agent(device=device,
                  state_size=state_size, action_size=action_size, random_seed=2, num_competing_agents=num_agents,
                  num_trainings_per_update=config.num_trainings_per_update,
                  time_steps_before_training=config.time_steps_before_training,
                  batch_size=config.batch_size,
                  num_episodes_to_increase_num_trainings=2000,
                  lr_actor=config.lr_actor,
                  lr_critic=config.lr_critic,
                  clip_grad_norm=config.clip_grad_norm,
                  replay_buffer=replay_buffer,
                  debug=False)

    scores_global, episode_durations, episode_timestep_reached = run_maddpg(agent=agent,
                                                                            n_episodes=1300,
                                                                            max_t=int(config.max_t),
                                                                            print_every=20,
                                                                            episodes_to_make_target_equal_to_local=config.episodes_to_make_target_equal_to_local)

    generate_training_plots(scores_global, episode_durations, episode_timestep_reached,
                            {"critic": agent.critics_local[0].__repr__(),
                             "actor": agent.actors_local[0].__repr__(),
                             "critic_optim": agent.critic_optimizers[0].__repr__().replace("\n", ", "),
                             "actor_optim": agent.actor_optimizers[0].__repr__().replace("\n", ", "),
                             "clip_grad_norm": agent.clip_grad_norm,
                             "batch_size": agent.batch_size,
                             "max_t": config.max_t,
                             "time_steps_before_training": agent.time_steps_before_training,
                             "num_trainings_per_update": agent.num_trainings_per_update,
                             "num_episodes_to_increase_num_trainings": agent.num_episodes_to_increase_num_trainings,
                             "noise_decay": agent.noise_decay,
                             "episodes_to_make_target_equal_to_local": config.episodes_to_make_target_equal_to_local
                             })
