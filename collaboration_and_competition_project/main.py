from collaboration_and_competition_project.ReplayBuffer import ReplayBuffer
from collaboration_and_competition_project.helpers import generate_training_plots

BUFFER_SIZE = int(1e5)  # replay buffer size # TODO: check value

from collaboration_and_competition_project import state_size, action_size, num_agents, device
from collaboration_and_competition_project.agent import Agent
from collaboration_and_competition_project.maddpg import run_maddpg

batch_size = 512

replay_buffer = ReplayBuffer(action_size=action_size,
                             buffer_size=BUFFER_SIZE,
                             batch_size=batch_size,
                             random_seed=2,
                             num_collaborating_agents=num_agents,
                             device=device)

agent = Agent(device=device,
              state_size=state_size, action_size=action_size, random_seed=2, num_competing_agents=num_agents,
              num_trainings_per_update=2,
              time_steps_before_training=5,
              batch_size=batch_size,
              num_episodes_to_increase_num_trainings=500,
              lr_actor=1e-3,
              lr_critic=1e-3,
              clip_grad_norm=True,
              weight_decay= 0.0001,
              start_noise_variance = 1.0,
              replay_buffer=replay_buffer,
              add_samples_only_if_high_reward=False,
              debug=False)

episodes_to_make_target_equal_to_local = 200
max_timesteps_per_episode = 500

scores_global, episode_durations, episode_timestep_reached = run_maddpg(agent=agent,
                                        n_episodes=1500,
                                        max_t=max_timesteps_per_episode,
                                        print_every=20,
                                        episodes_to_make_target_equal_to_local=episodes_to_make_target_equal_to_local)

details_dict = {"critic": agent.critics_local[0].__repr__(),
                 "actor": agent.actors_local[0].__repr__(),
                 "critic_optim": agent.critic_optimizers[0].__repr__().replace("\n", ", "),
                 "actor_optim": agent.actor_optimizers[0].__repr__().replace("\n", ", "),
                 "clip_grad_norm": agent.clip_grad_norm,
                 "batch_size": agent.batch_size,
                 "max_t": episodes_to_make_target_equal_to_local,
                 "time_steps_before_training": agent.time_steps_before_training,
                 "num_trainings_per_update": agent.num_trainings_per_update,
                 "num_episodes_to_increase_num_trainings": agent.num_episodes_to_increase_num_trainings,
                 "start_noise_variance": agent.start_noise_variance,
                 "noise_decay": agent.noise_decay,
                 "add_samples_only_if_high_reward": agent.add_samples_only_if_high_reward,
                 "episodes_to_make_target_equal_to_local": episodes_to_make_target_equal_to_local
                 }
generate_training_plots(scores_global, episode_durations, episode_timestep_reached,
                        details_dict)
