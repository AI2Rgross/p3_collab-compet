# Continous Control

## Introduction
In this project, I trained 2 agents to play tennis. Each agent controls a racket to hit a ball over the net. If the agent manage to put the ball over the net, it gets a reward of 0.1. But, if the ball hit the ground or goes out of bounds the rewards is -0.01. Therefore, the objective of each agent is to keep the ball in play as long as possible.

The observation space consists of 8 variables corresponding to the position and the velocity of the ball and the racket. Each agent receives its own observation.

There is only only two continous actions possible:
* moving toward or away of the net. 
* jumping

The task is episodic and in order to solve the environment, the agents must get an average score of +0.5 over 100 consecutive episodes.

The score for an agent is determined at the end of an episode. It corresponds to the sum of all the rewards without discounting obtained during the episode. 
To evaluate the average score, I only take into account the maximum score at each episode.

![Environment](/PNG/env.png)



## Download the main files
Download the following repo and unzip it: [click here](https://github.com/AI2Rgross/DRL/p3_collab_compet)

Also, download the right unity environment for your computer:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
 
Place the environment zip file in the p3_collab_compet folder and unzip it.


## Requirement
As I am working under ubuntu 16.04, I will give you a few tips to make it run smoothly. But, I recommand you to follow the instruction on the Udacity DRL github: https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continous_control/README.md. Especially, if you work under a different OS.

You can also have a look at: [Udacity] (https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md)
 

## Tips to prepare the environement
* Python 3
Check if you have a one of the following python versionn: 3.5.x or 3.6.x
If not, it s time to install it using the "sudo apt-get install" command.

* pip 3
You will also need to install pip for python 3. 
> sudo apt-get install python3-pip

* Pytorch:
check this link https://pytorch.org and get the command to install pytorch

* Ml agents for unity
First dowmload Ml agent ripo on https://github.com/Unity-Technologies/ml-agents.git

* Install ml agent:
> pip3 install .

if there is any trouble have a look at:
> https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

* Install unity
> pip3 install unity

* Install ipython
In order to run the notebook with python 3 also install ipyton.
> sudo pip3 install ipython


## Requirements for the collab-compet project
1. Go inside the collab-compet repo, and open a terminal in order to install the requirements:
> pip3 install -r requirements.txt*
Then, run the next command to open a notebook:
> ipython3 notebook*

2. Click on Collab_compet.ipynb to open it. Try to run the first cell where the libraries are imported to check if the UnityEnvironment is well installed.
Then, in the next cell correct the path of the **Tennis.x86_64** and try running it.

> env = UnityEnvironment(file_name=*"the/path/here/Tennis.x86_64"*)

Now everything is ready to either train your own solution or run the pre-computed solution.


## My files
Main files of the repository:

* The main part of the code (point for starting the environment, train the agent or test a solution) is in
*Collab_compet.ipynb* and *ihm_functions.py*

* The Agent class with DDPG, and othes basic functions to interact with the environment are in *agent.py*

* The Pytorch neural networks used to approximate the actor-critic functions used by each agent can be found in *model.py*

* The weights of the pytorch model for DDPG of the solved environment for the actor and critic:
	* checkpoint_actor.pth
	* checkpoint_critic.pth
 
* The weights of the pytorch model for Multi agents DDPG (without shared memory) of the solved environment for the actor and critic:
	* agent 0:
		* No_multi_checkpoint_actor_0.pth
		* No_multi_checkpoint_critic_0.pth
	* agent 1:
		* No_multi_checkpoint_actor_1.pth
		* No_multi_checkpoint_critic_1.pth

* The weights of the pytorch model for Multi agents DDPG (with shared memory) of the solved environment for the actor and critic:
	* agent 0:
		* W_multi_checkpoint_actor_0.pth
		* W_multi_checkpoint_critic_0.pth
	* agent 1:
		* W_multi_checkpoint_actor_1.pth
		* W_multi_checkpoint_critic_1.pth
        
* Installation notes and tips, brief description of the project: README.md

* Udacity original readme for instalation of the environment: Udacity_README.md

* My notes about DDPG: Report.pdf

The code I developped to solve the Tennis environment is based on code of the p2_continous_control and ddpg_bipedal exercices from the UDACITY DRL github available here: https://github.com/udacity/deep-reinforcement-learning.
