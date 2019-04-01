from typing import Callable

import torch
import torch.nn as nn

from torch.distributions import Normal, MultivariateNormal
from common.networks.initializer import init_linear_weights_xavier

class MLP(nn.Module):
    """Baseline of Multi-layer perceptron"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int = 0,
        output_activation: Callable = nn.Sequential(), # identity
        hidden_activation: Callable = nn.ReLU(),
    ):
        """Initialization with xavier

        Args:
            input_size (int): size of input layer
            output_size (int): size of output layer. if zero, it is not used
            hidden_sizes (list): sizes of hidden layers
            output_activation (function): activation function of output layer
            hidden_activation (function): activation function of hidden layers
            use_output_layer (bool): whether or not to use the last layer 
                                     for using subclass
        """

        super().__init__()

        self.fcs = nn.Sequential()

        # Hidden Layers
        # ========================================================================
        prev_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            self.fcs.add_module(f"hidden_fc{i}", nn.Linear(prev_size, next_size))
            self.fcs.add_module(f"hidden_fc_act{i}", hidden_activation)
            prev_size = next_size

        # Output Layers
        # ========================================================================
        if output_size:
            self.fcs.add_module(f"output", nn.Linear(prev_size, output_size))
            self.fcs.add_module(f"output_act", output_activation)
        
        self.apply(init_linear_weights_xavier)

    def forward(self, x: torch.Tensor):
        """Forward method implementation"""
        return self.fcs(x)

class NormalDistMLP(nn.Module):
    """ Multi-layer Perceptron with distribution output.(Gaussian, Normal)
        hidden layer size of mu, std is always same. 
        But it can be seperated or shared network    
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        mu_activation: Callable = nn.Sequential(), # identity
        sigma_activation: Callable = nn.Sequential(), # identity
        hidden_activation: Callable = nn.ReLU(),
        share_net: bool = True,
    ):
        '''Initialization with xavier

        Args:
            input_size (int): size of input layer
            output_size (int): size of output layer
            hidden_sizes (list): sizes of hidden layers
            mu_activation (function): activation function of mean(mu)
            sigma_activation (function): activation function of std or logstd(sigma)
            hidden_activation (function): activation function of hidden layers
            share_net (bool): whether using one network or sperate network
            dist_type (str): Select distribution type ('normal', 'gaussian')
        '''

        super().__init__()

        self.share_net = share_net

        if share_net:
            self.hidden_layer = MLP(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                hidden_activation=hidden_activation,
            )

            self.mu = nn.Sequential(
                                nn.Linear(hidden_sizes[-1], output_size),
                                mu_activation
                            )

            self.sigma = nn.Sequential(
                                nn.Linear(hidden_sizes[-1], output_size),
                                sigma_activation
                            )
        else:
            self.mu = MLP(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=hidden_sizes,
                output_activation=mu_activation,
                hidden_activation=hidden_activation
            )
            self.sigma = MLP(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=hidden_sizes,
                output_activation=sigma_activation,
                hidden_activation=hidden_activation
            )

        self.apply(init_linear_weights_xavier)
    
    def forward(self, x):
        if self.share_net:
            hidden_layer = self.hidden_layer.forward(x)
            
            mu = self.mu(hidden_layer)
            sigma = self.sigma(hidden_layer)
        else:
            mu = self.mu(x)
            sigma = self.sigma(x)

        return Normal(mu, sigma)
