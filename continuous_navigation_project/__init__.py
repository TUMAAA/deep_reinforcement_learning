from unityagents import UnityEnvironment

env = UnityEnvironment(file_name='continuous_navigation_project/Reacher_Twenty_Linux_NoVis/Reacher.x86_64')

print("\n\n---------------------\nEnvironment successfully started")
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
