# Project 1: Navigation

### Introduction

This is the report of Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)   Navigation project. 

For the environment we use the [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents) Banana environment.

The main idea is to approximate the Q function using a neural network.

## The environment

In the Reacher environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The table below depicts the environment:

![banana_env](images/reacher.gif)

### Reward

| event                      | reward  |
|----------------------------|:-------:|
| hand in global location    | +0.1    |
| hand out global location   | None    |

### State space

33-dimensional np-array.

### Actions

4 different continuous actions, corresponding to torque applicable to two joints.

### Solution

Average score of +30 over 100 consecutive episodes, and over all the 20 agents.

## Network architecture

The models are created at model.py, and in this project define by the following architectures:

    Actor network:
      input_dim: 33 (state size)
      hidden_1_size: 256
      activation_1: relu
      hidden_2_size: 128
      activation_2: relu
      output_dim: 4 (action size)


    Critic network:
      input_dim: 33 (state size)
      hidden_1_size: 260 (256 + action size)
      activation_1: relu
      hidden_2_size: 128
      activation_2: relu
      output_dim: 1

  note that the output of Critic network is a contiuous value, corresponding to the q-target.
  On hidden layer 1 we concatenate the actions, as is suggested in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).

## Agent

The agent class is in `agent.py`, wich mediates the interaction between the environment and the model.  

As we are using experience replay. We have a buffer to store te past episodes and rewards. The parameters for agent are:


| param        | vaule    | description 
|--------------|----------|--------------------------------------
|BUFFER_SIZE   | int(1e5) | replay buffer size
|BATCH_SIZE    | 128      | minibatch size
|GAMMA         | 0.99     | discount factor
|TAU           | 1e-3     | for soft update of target parameters
|LR (for both) | 1e-4     | learning rate 


## Training

Actor Critic methods are composed of four networks. The two main networks are the Actor and the Critic, the other two are copies of them, used to apply Fixed Q-Targets, like DQN method. Differently from this last one, Deep Deterministic Policy Gradient (DDPG) are efficient to deal with continuous space.

As on DQN, we have the learning and sampling steps.

* *Learning step*: The Actor networks learn which action take, or which contiuous value for each action, given a state. On the other hand, the Critic network receives the action taken by the Actor and the states to generete de expected q-value for it. So the Critic evaluates the state value function using TD estimate, calculates the Advantage Function and use this value to train the Actor, in the way that the Actor's goal is to maximize the Critic q-value.

* *Sampling step*:  the algorithm chooses an action A from a state s using a policy, observes the reward and next state and stores it on a buffer memory. On learning step we use this buffer to select samples of past episodes to compute its expected Q-values.

The DDPG algorithm is written bellow:

![ddpg_algo](images/ddpg.png)

To proceed with fixed Q-targets we use soft-updating, copying an amount of the local weights to the target weights. This update is given by:

θ_target = τ*θ_local + (1 - τ)*θ_target



## Results 

The following graph shows the results of the training run. Every blue dot in the graph represents the score of an
episode (drawn above the index of the episode). The orange dots are the rolling mean.

The problem was really solved at after episode 1357.

![scores graph](images/scores.png)

The weights learned during this run are saved in the files checkpoint_actor.pth and checkpoint_critic.pth.

## Future Works

For future works I would like to implement [AC3](https://arxiv.org/abs/1602.01783), and [D4PG](https://arxiv.org/abs/1804.08617), that uses multiple parallel copies of the same agent to distribute the task of generating experiences.

One other approach should be using Prioritized Experience Replay to avoid the cases where rare but important experiences could be missed or not so much used during learning.