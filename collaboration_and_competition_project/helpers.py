import numpy as np
from matplotlib import pyplot as plt


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