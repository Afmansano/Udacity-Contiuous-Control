import numpy as np
import random
import copy
import os

from collections import namedtuple, deque
from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


class ReplayMemory:

    def __init__(self, action_size, maxlen, batch_size, device, seed=42):
        self.action_size = action_size
        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.buff = deque(maxlen=maxlen)  # internal memory (deque)
        self.DEVICE = device
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        
    def add(self, state, action, reward, next_state, done):
        '''add the event to the memory buffer'''
        experience = self.experiences(state, action, reward, next_state, done)
        self.buff.append(experience)
        
    
    def sample(self, with_replacement=True):    
        '''randomly sample events from memory'''
        experiences = random.sample(self.buff, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences 
                                             if e is not None])).float().to(self.DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences 
                                              if e is not None])).float().to(self.DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences 
                                              if e is not None])).float().to(self.DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences 
                                                  if e is not None])).float().to(self.DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences 
                                            if e is not None]).astype(np.uint8)).float().to(self.DEVICE)
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buff)
        

class Agent(): 
    
    def __init__(self, env, buffer_size, batch_size, gamma, 
                 TAU, lr_actor, lr_critic, weight_decay, actor_layers, critic_layers, seed=42):        

        # get info from environment
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        state_size = len(state)
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
    
        self.env = env
        self.brain_name = brain_name
        self.n_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Hyperparameters
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Running on {0}...'.format(self.DEVICE))

        '''Actor'''
        self.local_actor = Actor(state_size, action_size, actor_layers, seed).to(self.DEVICE)
        self.target_actor = Actor(state_size, action_size, actor_layers, seed).to(self.DEVICE)
        self.actor_optim = optim.Adam(self.local_actor.parameters(), lr=self.LR_ACTOR)
        
        '''Critic'''
        self.local_critic = Critic(state_size, action_size, critic_layers, seed).to(self.DEVICE)
        self.target_critic = Critic(state_size, action_size, critic_layers, seed).to(self.DEVICE)
        self.critic_optim = optim.Adam(self.local_critic.parameters(), 
                                       lr=self.LR_CRITIC, 
                                       weight_decay=self.WEIGHT_DECAY)
            
        
        #add the noise process
        self.noise = Noise((self.n_agents, self.action_size), seed=seed)
        
        #replay buffer
        self.memory = ReplayMemory(self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.DEVICE, seed)
        
        
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for i in range(self.n_agents):
            self.memory.add(states[i, :], actions[i, :], rewards[i], next_states[i, :], dones[i])

        # Learn 
        if len(self.memory) > self.BATCH_SIZE:
            states, actions, rewards, next_states, dones = self.memory.sample()
            self.learn(states, actions, rewards, next_states, dones)
            
        
    def act(self, state, add_noise=True):
        '''Returns actions for states and policies'''
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.local_actor.eval()
        
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    
    def learn(self, states, actions, rewards, next_states, dones):
        
        '''Critic'''
        # get next-state and action from targe'
        actions_next = self.target_actor(next_states)
        # get Q-target for next states
        Q_target_next = self.target_critic(next_states, actions_next)
        # get Q-target for current states
        Q_targets = rewards + (self.GAMMA * Q_target_next * (1-dones))
        # loss
        Q_expected = self.local_critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        '''Actor'''
        #loss
        actions_pred = self.local_actor(states)
        actor_loss = -self.local_critic(states, actions_pred).mean() # to maximization
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        self.soft_update(self.local_critic, self.target_critic)
        self.soft_update(self.local_actor, self.target_actor)
        
        self.actor_loss = actor_loss.data
        self.critic_loss = critic_loss.data 
    
    def soft_update(self, local_model, target_model):
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
    def train(self, n_episodes):
        scores = []
        last_scores = deque(maxlen=100)

        for episode in range(n_episodes):
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            self.reset()
            episode_scores = np.zeros(self.n_agents)

            while True:
                actions = self.act(states)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                self.step(states, actions, rewards, next_states, dones)
                episode_scores += rewards
                states = next_states

                if np.any(dones):
                    break

            mean_scores = np.mean(episode_scores)
            scores.append(mean_scores)
            last_scores.append(mean_scores)
            last_scores_mean = np.mean(last_scores)
            print('\rEpisode: \t{} \tScore: \t{:.2f} \tMean Scores: \t{:.2f}'.format(episode, mean_scores, last_scores_mean), end="")  

            if last_scores_mean >= 30.0:
                print('\nEnvironment solved in {:d} episodes!\tMean Scores: {:.2f}'.format(episode, last_scores_mean))
                self.save()
                break 

        return scores
                
            
    def reset(self):
        self.noise.reset()
        
        
    def save(self):
        torch.save(self.local_actor.state_dict(), 'checkpoint_actor.pth')      
        torch.save(self.local_critic.state_dict(), 'checkpoint_critic.pth') 
        

class Noise():
    """Ornstein-Uhlenbeck process."""
    """copied from https://github.com/marcelloaborges/Reacher-Continuous-Control/blob/master/ddpg_agent.py"""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, seed=42):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = np.ones(size)*mu
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        
    
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state