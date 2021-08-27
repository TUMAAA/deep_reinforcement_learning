[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

#### Environment
This project uses the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

#### Multi-Agent
The environment is also available as multi-agent environment where in each time step 20 different agents can be interacted with.
The rewards over time steps are accumulated per agent. 
We use this version (as opposed to the single agent environment) to solve the challenge.

#### Goal of the Challenge

The trained agent must get an average score of +30 over 100 consecutive episodes.
In case multiple agents are used (the multi-agent environment) the score per episode is the mean score over all agents. 

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


## Getting started

Detailed instructions to get the environment properly set up with the right Python version and dependencies is provided in the Jupyter notebook Continuous_Navigation.ipynb.
They include setting up the python environment with dependencies and the Unity environment. 

## Instructions
Make sure to run the Jupyter Notebook server from the repository root. This way the needed packaged "module" can be imported.

To train the agent simply follow the instructions inside the Continuous_Navigation.ipynb.

## Sample output
The Report.html is a report showing the results obtained when solving the challenge.

## Implemented Solution and Lessons Learnt
We use the multi-agent environment explained above. However, there is only one policy trained which interacts with all environments to collect experiences.

The algorithm is described in the pseudo code below

![pseudo-code](algorithm.png)

while having the following minor modification:
- In each time step all 20 agents update the experience buffer.
- After every time_steps_before_training (in solution set to 20) we perform num_trainings_per_update (in solution set to 20) backpropagation passes to update the actor and critic.
  
In solving the challenge different ideas were tested. They are listed below:

|Num| Idea | Rationale |Usefulness |
|---|-----|--------------|----------|
|1|Setting target equal to local model (actor and critic) after every **episodes_to_make_target_equal_to_local** episodes|The soft update does not ensure that over long time the target becomes equal to the local model. In DQN this is usually done.| No significant improvement or degradation|
|2|Resetting the experience replay buffer every pre-defined number of episodes | Since the replay buffer is fairly large it will have very outdated samples in it. The idea is that as training progress we get new experience in. This inspired by human learning where a human learns more from recent experiences.| No significant improvement or degradation |
|3|Increasing the number of repeated training iterations (forward-backward passes) every **num_episodes_to_increase_num_trainings** episodes|As the training progresses and the agent becomes more confident (and the learning curve stable) it can increase its learning rate| No significant improvement to learning while significantly increasing the learning time|

For idea 3 it is worth mentioning that this also ensures setting the target model params equal to the local params at initialization.
This step is part of the original DDPG algorithm and forgotten in the reference code by Udacity as nicely pointed out by one of the course participants in the [forum](https://knowledge.udacity.com/questions/98687).

We also implemented one policy per agent. The implementation was not tested thoroughly. It is in the branch multi_agent_multi_policy
 
## Final notes
The submitted code builds upon the baseline notebooks and python files provided by Udacity.