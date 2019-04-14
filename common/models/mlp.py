

import torch
import torch.nn as nn
from typing import Callable, Union, List, Tuple
from torch.distributions import Normal, Categorical

from common.models.utils import init_linear_weights_xavier

class MLP(nn.Module):
    """Baseline of Multi-layer perceptron"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int = 0,
        hidden_activation: Callable = nn.ReLU(),
        output_activation: Callable = nn.Sequential(), # identity
    ):
        """Initialization with xavier

        Args:
            input_size: The size of input layer
            output_size: The size of output layer. if 0, not used
            hidden_sizes: The sizes of hidden layers 
                        (e.g) 1. [] : no hidden layers
                              2. [128, 256] : first 128 layers 
                                             second 256 layers
            hidden_activation: The activation function of hidden layers
            output_activation: The activation function of output layer.
                               This shape must be same with output_size above
            
        """
        super().__init__()

        self.fcs = nn.Sequential()

        # Hidden Layers
        # ===========================================================================
        prev_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            self.fcs.add_module(f"hidden_fc{i}", nn.Linear(prev_size, next_size))
            self.fcs.add_module(f"hidden_fc_act{i}", hidden_activation)
            prev_size = next_size

        # Output Layers
        # ===========================================================================
        if output_size > 0:
            self.fcs.add_module(f"output{i}", nn.Linear(prev_size, output_size))
            self.fcs.add_module(f"output_act{i}", output_activation)

        # Initialized
        self.apply(init_linear_weights_xavier)

    def forward(self, x: torch.Tensor):
        """Forward method implementation"""       

        return self.fcs(x)

class SepActorCritic(nn.Module):
    ''' Actor Critic model with seperate networks. '''
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        actor_output_size: int,
        critic_output_size: int,
        dist: Union[Normal, Categorical],
        hidden_activation: Callable = nn.ReLU(),
        actor_output_activation: Callable = nn.Sequential(), # identity
        critic_output_activation: Callable = nn.Sequential(), # identity
    ):
        """Initialization with xavier

        Args:
            input_size: The size of input layer
            hidden_sizes: The sizes of hidden layers 
                        (e.g) 1. [] : no hidden layers
                              2. [128, 256] : first 128 layers 
                                             second 256 layers
            actor_output_size: The size of Actor output layer. 
            critic_output_size: The size of Critic output layer.
            dist: an ouptut distribution of Actor, Normal or Categorical
            hidden_activation: The activation function of hidden layers
            actor_output_activation: The activation function of output layer.
                               This shape must be same with actor_output_size above
            critic_output_activation: The activation function of output layer.
                               This shape must be same with critic_output_size above
            
        """
        super().__init__()
        if dist == Categorical:
            assert type(actor_output_activation) == type(nn.Softmax()), \
                   "If you use Categorical, output activation must be softmax"
        
        self.dist = dist

        self.critic = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=critic_output_size,
            hidden_activation=hidden_activation,
            output_activation=critic_output_activation
        )
        
        self.actor = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=actor_output_size,
            hidden_activation=hidden_activation,
            output_activation=actor_output_activation
        )
        self.apply(init_linear_weights_xavier)

    def forward(self, x):
        ac = self.actor(x)
        value = self.critic(x)

        if self.dist == Normal: 
            std = (torch.ones_like(ac) * self.std).exp()
            dist = Normal(ac, std)
            
            return dist, value
        elif self.dist == Categorical:
            dist = Categorical(ac)
            
            return dist, value
        else: # ac = policy
            return ac, value

class SharedActorCritic(MLP):
    ''' Actor Critic model with a share network. '''
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        actor_output_size: int,
        critic_output_size: int,
        dist: Union[Normal, Categorical],
        hidden_activation: Callable = nn.ReLU(),
        actor_output_activation: Callable = nn.Sequential(), # identity
        critic_output_activation: Callable = nn.Sequential(), # identity
    ):
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=0,
            hidden_activation=hidden_activation,            
        )
        if dist == Categorical:
            assert type(actor_output_activation) == type(nn.Softmax()), \
                   "If you use Categorical, output activation must be softmax"

        self.dist = dist
        self.actor = nn.Sequential()
        self.critic = nn.Sequential()

        last_size = hidden_sizes[-1] if hidden_sizes else input_size

        self.actor.add_module("actor", nn.Linear(last_size, actor_output_size))
        self.actor.add_module("actor_act", actor_output_activation)

        self.critic.add_module("critic", nn.Linear(last_size, critic_output_size))
        self.critic.add_module("critic_act", critic_output_activation)
        
        self.apply(init_linear_weights_xavier)

    def forward(self, x):
        hidden = self.fcs(x)

        ac = self.actor(hidden)
        value = self.critic(hidden)
        
        if self.dist == Normal: 
            std = (torch.ones_like(ac) * self.std).exp()
            dist = Normal(ac, std)
            
            return dist, value
        elif self.dist == Categorical:
            dist = Categorical(ac)
            
            return dist, value
        else: # ac = policy
            return ac, value