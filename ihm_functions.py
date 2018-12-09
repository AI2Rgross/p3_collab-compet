from unityagents import UnityEnvironment
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
from agent import Agent

def plot_score(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    
    
    
def Load_environment(file_name,train_mode):
    # Load environment
    env = UnityEnvironment(file_name)

    # Define brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Number of agents 
    num_agents = len(env_info.agents)

    # Size of each action
    action_size = brain.vector_action_space_size

    # Examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    
    return  env,brain_name,num_agents,action_size,state_size


########################
## TRAINING FUNCTIONS ##
########################
# Training function for one agent using the ddpg method. One agent play on both side of the tennis court.

def ddpg(env, state_size, action_size, brain_name,num_agents, agent,rate=0.95,n_episodes=5000, max_t=100000):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations # get the current state (for each agent)
        agent.reset() # reset the noise added to the state. Makes the training more robust.
        score=np.zeros(num_agents) # initialize the score (for each agent)
        for t in range(max_t):
            actions = [agent.act(states[i],rate=rate) for i in range(num_agents)] # get action from each agent based on the current state
            env_info = env.step(actions)[brain_name]  # update environment informations with the actions of each agent
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            score = score+rewards  # update the score for each agent
            dones = env_info.local_done  # see if episode is finished
            agent.step(states, actions, rewards, next_states, dones,num_agents,2) # add (state,actions,rewards,next_states) to replay buffer and train the actor-critic neural network
            states = next_states # roll over the state to next time step
            if any(dones):
                break
                
        scores.append(np.max(score)) # add maximum score for to display the results later
        scores_deque.append(np.max(score)) # add maximum score to windows for convergence check
        print('\rEpisode {}\tAverage Score: {}\tScore: {}'.format(i_episode, np.mean(scores_deque), np.max(score)), end="")
        if i_episode>100 and np.mean(scores_deque)>0.5: # check if the env is solved
            print("envionment solved after {}\n", i_episode)
            torch.save(agent.actor_local.state_dict(), 'ddpg_checkpoint_actor.pth') # save actor weights
            torch.save(agent.critic_local.state_dict(), 'ddpg_checkpoint_critic.pth')# save critic weights
            return scores
        
        if i_episode%100 ==0: # save intermediary solution every 100 iter
            torch.save(agent.actor_local.state_dict(), 'ddpg_checkpoint_actor.pth') # save actor weights
            torch.save(agent.critic_local.state_dict(), 'ddpg_checkpoint_critic.pth') # save critic weights
   
    return scores


# Training function for two agents using the ddpg method. 

def multi_ddpg(env,\
               state_size,\
               action_size,\
               brain_name,\
               num_agents,\
               agent,\
               n_episodes=5000,\
               max_t=100000,\
               weight_name="checkpoint",\
               rate=0.95,\
               mode=1):
    
    scores_deque = deque(maxlen=100 )
    scores = []

    for i_episode in range(1, n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        score=np.zeros(num_agents)  # initialize the score (for each agent)

        for i in range(num_agents):
            agent[i].reset() # reset the noise added to the state. Makes the training more robust.
            
        for t in range(max_t):
           # actions = agent.act(states)
            actions = [agent[i].act(states[i],add_noise=True,rate=rate) for i in range(num_agents)] # get action from each agent based on the current state
            env_info = env.step(actions)[brain_name]  # update environment informations with the actions of each agent
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            score = score+rewards  # update the score for each agent
            dones = env_info.local_done  # see if episode finished
            # agent[i].step: add (states,actions,rewards,next_states) to replay buffer of each agent 
            # train the actor critic Neural Network of each agent
            # each agent share the same information
            # There are two possibilities, mode = 0 for not sharing (state,action, etc) and mode = 1 for sharing (state,action, etc). In the second case, the agents are trained to play on both side of the tennis field.
            if(mode==0):
              [agent[i].step(states, actions, rewards, next_states, dones,num_agents,i) for i in range(num_agents)]
            else:
              [agent[i].step(states, actions, rewards, next_states, dones,num_agents,2) for i in range(num_agents)]
            states = next_states # roll over the state to next time step
            if any(dones):
                break
                
        scores.append(np.max(score)) # save the best agent score for display
        scores_deque.append(np.max(score)) # save the best agent score into the windows for convergence checking
        print('\rEpisode {}\tAverage Score: {}\t max Score: {}'.format(i_episode, np.mean(scores_deque), np.max(score)), end="")
        
        if i_episode%100 ==0 or  i_episode>100 and np.mean(scores_deque)>0.5:
            [torch.save(agent[i].actor_local.state_dict(), weight_name+'_actor'+str(i)+'.pth') for i in range(num_agents)] # save actor weights for each agents
            [torch.save(agent[i].critic_local.state_dict(), weight_name+ '_critic'+str(i)+'.pth') for i in range(num_agents)] # save critic weights for each agents
            
        if i_episode>100 and np.mean(scores_deque)>0.5: # check if env is solved
            print("envionment solved after {}\n", i_episode)
            return scores

    return scores



#######################
## TESTING FUNCTIONS ##
#######################
# Testing function for one agent.
def ddpg_Test(env,brain_name,num_agents,agent,file_name_actor,file_name_critic,n_episodes=5, max_t=100000):
    """ Visualize agent using saved checkpoint. """
    # load saved weights
    agent.actor_local.load_state_dict(torch.load(file_name_actor))
    agent.critic_local.load_state_dict(torch.load(file_name_critic)) 
    scores = []  # list containing scores from each episode
 
    for i_episode in range(1, n_episodes):
        env_info = env.reset(train_mode=False)[brain_name]  # environment informations
        states = env_info.vector_observations 
        agent.reset()
        score = np.zeros(num_agents) 
        for t in range(max_t):
            actions = agent.act(states,add_noise=False) # get action using the DDPG algorythme (for each agent)
            env_info = env.step(actions)[brain_name]  # update environment informations with the actions of each agent
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            states = next_states  # roll over the state to next time step
            score += rewards  # update the score for each agent
            if any(dones):  # see if episode has finished
                break
        scores.append(np.max(score)) # save the best score between both agents
        print('\rEpisode {}\tAverage Score: {}'.format(i_episode, np.max(score)), end="")

    return scores

# Testing function for two agent.
# agent.act is definied in agent.py

def double_ddpg_Test(env,brain_name,num_agents,agent,file_name_actor,file_name_critic,n_episodes=5000, max_t=10000000):
    """ Visualize agent using saved checkpoint. """
    # load saved weights
    for i in range(num_agents):
        agent[i].actor_local.load_state_dict(torch.load(file_name_actor[i]))
        agent[i].critic_local.load_state_dict(torch.load(file_name_critic[i])) 
        
    scores = []                        # list containing scores from each episode
    score = 0
 
    for i_episode in range(1, n_episodes):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations 
        [agent[i].reset() for i in range(num_agents)]
        score = np.zeros(num_agents) 
        for t in range(max_t):
            actions = [agent[i].act(states[i],add_noise=False) for i in range(num_agents)] # get action using the DDPG algorythme (for each agent)
            env_info = env.step(actions)[brain_name]  # update environment informations with the actions of each agent
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            states = next_states
            score =score+ rewards
            if any(dones):
                break
        scores.append(np.max(score)) # save the best score between both agents
        print('\rEpisode {}\tAverage Score: {}'.format(i_episode, np.max(score)), end="")
    return scores






























# multi agent with no shared memory
def multi_ddpg_ref(env, env_info, state_size, action_size, brain_name,num_agents, agent,n_episodes=5000, max_t=1000000):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        for i in range(num_agents):
            agent[i].reset() # reset the noise added to the state. Makes the training more robust.
        score=np.zeros(num_agents)  # initialize the score (for each agent)
        for t in range(max_t):
           # actions = agent.act(states)
            actions = [agent[i].act(states[i],rate=0.95) for i in range(num_agents)] # get action from each agent based on the current state
            env_info = env.step(actions)[brain_name]  # update environment informations with the actions of each agent
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            score = score+rewards  # update the score for each agent
            dones = env_info.local_done  # see if episode finished
            # agent[i].step: add (state,action,reward,next_state) to replay buffer of each agent 
            # train the actor critic Neural Network of each agent
            #  agent do not share the same information

            [agent[i].step(states, actions, rewards, next_states, dones,num_agents,i) for i in range(num_agents)]
            states = next_states # roll over the state to next time step
            if any(dones):
                break
                
        scores.append(np.max(score)) # save the best agent score for display
        scores_deque.append(np.max(score)) # save the best agent score into the windows for convergence checking
        print('\rEpisode {}\tAverage Score: {}\tScore: {}'.format(i_episode, np.mean(scores_deque), np.max(score)), end="")
        if i_episode>100 and np.mean(scores_deque)>0.5: # check if env is solved
            print("envionment solved")
            [torch.save(agent[i].actor_local.state_dict(), 'No_multi_checkpoint_actor'+str(i)+'.pth') for i in range(num_agents)] # save actor weights for each agents
            [torch.save(agent[i].critic_local.state_dict(), 'No_multi_checkpoint_critic'+str(i)+'.pth') for i in range(num_agents)] # save critic weights for each agents
            return scores
        
        if i_episode%100 ==0:
            [torch.save(agent[i].actor_local.state_dict(), 'No_multi_checkpoint_actor'+str(i)+'.pth') for i in range(num_agents)] # save actor weights for each agents
            [torch.save(agent[i].critic_local.state_dict(), 'No_multi_checkpoint_critic'+str(i)+'.pth') for i in range(num_agents)] # save critic weights for each agents
   
    return scores

# multi agent shared memory
def multi_ddpg_ref2(env, env_info, state_size, action_size, brain_name,num_agents, agent,n_episodes=5000, max_t=100000):
    scores_deque = deque(maxlen=100, weight_name="checkpoint")
    scores = []

    for i_episode in range(1, n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        for i in range(num_agents):
            agent[i].reset() # reset the noise added to the state. Makes the training more robust.
        score=np.zeros(num_agents)  # initialize the score (for each agent)
        for t in range(max_t):
           # actions = agent.act(states)
            actions = [agent[i].act(states[i],rate=0.95) for i in range(num_agents)] # get action from each agent based on the current state
            env_info = env.step(actions)[brain_name]  # update environment informations with the actions of each agent
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            score = score+rewards  # update the score for each agent
            dones = env_info.local_done  # see if episode finished
            # agent[i].step: add (states,actions,rewards,next_states) to replay buffer of each agent 
            # train the actor critic Neural Network of each agent
            # each agent share the same information
            [agent[i].step(states, actions, rewards, next_states, dones,num_agents,2) for i in range(num_agents)]
            states = next_states # roll over the state to next time step
            if any(dones):
                break
                
        scores.append(np.max(score)) # save the best agent score for display
        scores_deque.append(np.max(score)) # save the best agent score into the windows for convergence checking
        print('\rEpisode {}\tAverage Score: {}\tScore: {}'.format(i_episode, np.mean(scores_deque), np.max(score)), end="")
        if i_episode>100 and np.mean(scores_deque)>0.5: # check if env is solved
            print("envionment solved")
            [torch.save(agent[i].actor_local.state_dict(), weight_name+'_actor'+str(i)+'.pth') for i in range(num_agents)] # save actor weights for each agents
            [torch.save(agent[i].critic_local.state_dict(),weight_name+ '_critic'+str(i)+'.pth') for i in range(num_agents)] # save critic weights for each agents
            return scores
        
        if i_episode%100 ==0:
            [torch.save(agent[i].actor_local.state_dict(), 'W_multi_checkpoint_actor'+str(i)+'.pth') for i in range(num_agents)] # save actor weights for each agents
            [torch.save(agent[i].critic_local.state_dict(), 'W_multi_checkpoint_critic'+str(i)+'.pth') for i in range(num_agents)] # save critic weights for each agents
   
    return scores


