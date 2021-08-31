[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting started

Follow these instructions if you are working locally (you want to setup a Jupyter Notebook locally).

#### Setup the right Python version
Use Python 3.6. You may want to either use a virtual env or setup a Conda environment as explained here:

https://github.com/udacity/deep-reinforcement-learning#dependencies

#### Get the python dependencies requirements.txt
Checkout the ./python folder of the Udacity DRLND repo:

https://github.com/udacity/deep-reinforcement-learning

It has a requirements.txt to ensure installing the necessary dependencies in the version specified by the course
(do this only when you want to work locally). Otherwise use the reference notebook provided by the course which is pre setup properly.

#### Get the Pre-built Unity Environment
Since this is an RL problem an environment is needed.
The course requires you to use the pre-built environment.
Get the right version for you from the link provided in [Tennis_Linux_NoVis/README](Tennis_Linux_NoVis/README) and [Tennis_Linux/README](Tennis_Linux/README).
The latter is recommended as you may need to work with the visual environment to understand how your agents 
are interacting with the environment. 


## Instructions
Simply run main.py
`
python -m collaboration_and_competition.main
`


## Sample output
The Report.pdf is a report showing the results obtained when solving the challenge.
The trained agent checkpoint files are found under [trained_models/](trained_models/).
The specific files with which the challenge was solved are mentioned in the report.

## Implemented Solution and Lessons Learnt
We use the multi-agent environment explained above. However, there is only one policy trained which interacts with all environments to collect experiences.

A detailed treatment of the algorithm is provided in the report [file](Report.pdf). 

### Ideas for the Future
Ideas for the future are furthermore discussed in the report file.
 
## Final notes
Some of this submitted code builds upon the baseline notebooks and python files provided by Udacity.