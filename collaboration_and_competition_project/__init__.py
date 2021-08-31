import os
import pathlib

import torch
from unityagents import UnityEnvironment

# Necessary to allow working from cmd line terminal as well (pycharm handles that by itself through setting
# current workdir to the parent folder of the script)
module_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(module_dir)

print("----------------CWD: {}".format(os.getcwd()))
USE_VIS=False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if USE_VIS:
    env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')
else:
    env = UnityEnvironment(file_name='Tennis_Linux_NoVis/Tennis.x86_64')

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

EPISODE_LENGTH_FOR_AVERAGING = 100  # num episodes over which score avg should be > MIN_AVG_SCORE_OVER_LAST_HUNDRED_EPISODES_TO_BEAT
MIN_AVG_SCORE_OVER_LAST_HUNDRED_EPISODES_TO_BEAT = 0.5
TRAINED_MODELS_DIR = "trained_models"
MADDP_MODEL_SUBDIR="maddpg"
DDP_MODEL_SUBDIR="ddpg"