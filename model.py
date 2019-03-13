import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def init_layers(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    '''Actor network'''
    
    def __init__(self, state_size, action_size, hidden_layers, seed=42):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer_sizes = [state_size] + hidden_layers
        layer_dims = zip(layer_sizes[:-1], layer_sizes[1:])
        self.hidden_layers = nn.ModuleList([nn.Linear(h1, h2) 
                                           for h1, h2 in layer_dims])
        self.output = nn.Linear(layer_sizes[-1], action_size)
        self.initialize_weights()
        
        
    def initialize_weights(self):
        for layer_ in self.hidden_layers:
            layer_.weight.data.uniform_(*init_layers(layer_))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        
    
    def forward(self, state):
        '''maps state -> action'''
        x = state
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return torch.tanh(self.output(x))
    
    
class Critic(nn.Module):
    '''Critic network'''
    
    def __init__(self, state_size, action_size, hidden_layers, seed=42):
        hidden_layers = list(hidden_layers) #makes a copy, so the original will not be changed
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        layer_sizes = [state_size] + hidden_layers
        layer_dims = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        layer_dims[1] = (layer_dims[1][0] + action_size, layer_dims[1][1]) # add actions on first hidden layer 
        self.hidden_layers = nn.ModuleList([nn.Linear(h1, h2) 
                                           for h1, h2 in layer_dims])
        self.output = nn.Linear(layer_sizes[-1], 1)
        self.initialize_weights()     
        

    def initialize_weights(self):    
        for layer_ in self.hidden_layers:
            layer_.weight.data.uniform_(*init_layers(layer_))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
    
    
    def forward(self, state, action):
        '''maps (state, action) -> q-value'''
        x = state
        for i, _layer in enumerate(self.hidden_layers):
            if i == 1:
                x = torch.cat((x, action), dim=1)
            x = F.relu(_layer(x))
        return self.output(x)
