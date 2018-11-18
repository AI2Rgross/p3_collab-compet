# Continous Control

## Introduction
In this project, I trained 2 agents to play tennis. Each agent controls a rackets to hit a ball over the net. If the agent manage to do put the ball over the net, it get a reward of 0.1. But, the ball hit the ground or goes out of bounds the rewards is -0.01. Therefore, the objective of each agent is to keep the ball in play.


The observation space consists of 8 variables corresponding to position and velocity of the ball and the racket. Each agent receives its own observation.
There is only only two continous action possible:
- moving toward or way of the net. 
- jumping
The task is episodic and in order to solve the environment, the agents must get an average score of +0.5 over 100 consecutive episodes.

The score for an agent is determined at the end of an episode. It corresponds to the sum of all the rewards without discounting. 
To evaluate the average score, I take only into account the maximum score at each episode.



## Download the main files
Download the repo and unzip it:
https://github.com/AI2Rgross/DRL/p3_collab_compet

Download the unity environment:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
 

Place the zip file in the p3_collab_compet folder and unzip the file.


## Requirement
As I am working under ubuntu 16.04, I will give you a few tips to make it run smoothly. But, I recommand you to follow the instruction on the Udacity DRL github: https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continous_control/README.md. Especially, if you work under a different OS.

You can also have a look at:
https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md
 

## Tips to prepare the environement
- Python 3
Check if you have a one of the following python versionn: 3.5.x or 3.6.x
If not, it s time to install it using the "sudo apt-get install" command.

- pip 3
You will also need to install pip for python 3. 
sudo apt-get install python3-pip

- Pytorch:
check this link https://pytorch.org and get the command to install pytorch

- Ml agents for unity
First dowmload Ml agent ripo on https://github.com/Unity-Technologies/ml-agents.git

- Install ml agent:
pip3 install .

if there is any trouble have a look at:
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

- Install unity
pip3 install unity

- ipython
In order to run the notebook with python 3 also install ipyton.
sudo pip3 install ipython[all]


## Requirements for the collab-compet project
Go inside the collab-compet repo, and open a terminal in order to install the requirements:
pip3 install -r requirements.txt
Then run the next command to open a notebook:
ipython3 notebook

click on Collab_compet.ipynb to open it.
Try to run the first cell where the libraries are imported to check if UnityEnvironment is well installed.

Then, in the next cell correct the path of the Tennis.x86_64 and try running it. Be careful to use the multi-agent environment.

env = UnityEnvironment(file_name="the/path/here/Tennis.x86_64")

Now everything is ready to either train your own solution or run the pre-computed solution.


## My files
Main files of the repository:

    - The main part of the code: point for starting the environment, train the agent or test a solution.
        Collab_compet.ipynb

    - the Agent class with DDPG, and othes basic functions to interact with the environment.
        agent.py

    - The Pytorch neural networks used to approximate the actor-critic functions used by each agent.
        model.py

    - the weights of the pytorch model for DDPG of the solved environment for the actor and critic.
        checkpoint_actor.pth
        checkpoint_critic.pth 
        
    - Installation notes and tips, brief description of the project
        README.md

    - Udacity original readme for instalation of the environment.
        Udacity_README.md

    - My notes about DDPG
        Report.pdf

The code I developped to solve the Tennis environment is based on code of the p2_continous_control and ddpg_bipedal exercices from the UDACITY DRL github available here: https://github.com/udacity/deep-reinforcement-learning.
