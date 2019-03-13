# Project 2: Continuous Control
Continuous Control
### Introduction

This is the impelementation of Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) Continuous Control project. 

For the environment we use the [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents) Reacher environment.

In the Reacher environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

In this project we are using the 2 agent environment.

![reacher_env](images/reacher.gif)

The table below depicts the environment:


#### State Space

The environment state space is composed of a 33-dimensional np-array.:

```python
States look like: [  0.00000000e+00  -4.00000000e+00   0.00000000e+00   1.00000000e+00
  -0.00000000e+00  -0.00000000e+00  -4.37113883e-08   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00  -1.00000000e+01   0.00000000e+00
   1.00000000e+00  -0.00000000e+00  -0.00000000e+00  -4.37113883e-08
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   5.75471878e+00  -1.00000000e+00
   5.55726624e+00   0.00000000e+00   1.00000000e+00   0.00000000e+00
  -1.68164849e-01]
```


#### Action Space

For this environment we have 4 different continuous actions, corresponding to torque applicable to two joints.


#### Solving the environment

As we are using 20 agents environment, we have to take into account the presence of many agents. The agents must get an average score of +30 (over 100 consecutive episodes, and over all the 20 agents).


## Getting started

### Installation requirements

This project was developed using Udacity workspace. If you are using this environment just need to run the command:

!pip -q install ./python

Otherwise, if you are running locally, follow the instructions bellow:

- You first need to configure a Python 3.6 / PyTorch 0.4.0 environment with the needed requirements as described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)

- Then you have to install the Unity environment The environment can be downloaded from one of the links below for all operating systems:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


### Train the agent
    
Execute the Continuous_Control.ipynb notebook within this Nanodegree Udacity Online Workspace for "project #2  Continuous Control" (or build your own local environment and make necessary adjustements for the path to the UnityEnvironment in the code )
