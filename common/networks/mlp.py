from typing import Callable

import torch
import torch.nn as nn

from torch.distributions import Normal
from common.networks.initializer import init_linear_weights_xavier


class MLP(nn.Module):
    """ Multi-layer Perceptron """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = nn.ReLU(),
        output_activation: Callable = nn.Sequential(),  # identity
    ):
        super(MLP, self).__init__()

        self.fcs = nn.Sequential()

        # Hidden Layers
        prev_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            self.fcs.add_module(f"fc_{i}", nn.Linear(prev_size, next_size))
            self.fcs.add_module(f"fc_act_{i}", hidden_activation)
            prev_size = next_size

        # Output Layers
        self.fcs.add_module("output", nn.Linear(prev_size, output_size))
        self.fcs.add_module("output_act", output_activation)

        # Initialize weights and biases
        self.apply(init_linear_weights_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcs(x)


class DistributedMLP(nn.Module):
    """ Multi-layer Perceptron """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        mu_hidden_sizes: list,
        sigma_hidden_sizes: list,
        mu_hidden_activation: Callable = nn.ReLU(),
        sigma_hidden_activation: Callable = nn.ReLU(),
        mu_output_activation: Callable = nn.Sequential(),  # identity
        sigma_output_activation: Callable = nn.Softplus(),
    ):
        super(DistributedMLP, self).__init__()

        self.mu_fcs = nn.Sequential()

        # Mu Hidden Layers
        prev_size = input_size
        for i, next_size in enumerate(mu_hidden_sizes):
            self.mu_fcs.add_module(f"fc_{i}", nn.Linear(prev_size, next_size))
            self.mu_fcs.add_module(f"fc_act_{i}", mu_hidden_activation)
            prev_size = next_size

        # Mu Output Layers
        self.mu_fcs.add_module("output", nn.Linear(prev_size, output_size))
        self.mu_fcs.add_module("output_act", mu_output_activation)

        self.sigma_fcs = nn.Sequential()

        # Sigma Hidden Layers
        prev_size = input_size
        for i, next_size in enumerate(sigma_hidden_sizes):
            self.sigma_fcs.add_module(f"fc_{i}",
                                      nn.Linear(prev_size, next_size))
            self.sigma_fcs.add_module(f"fc_act_{i}", sigma_hidden_activation)
            prev_size = next_size

        # Mu Output Layers
        self.sigma_fcs.add_module("output", nn.Linear(prev_size, output_size))
        self.sigma_fcs.add_module("output_act", sigma_output_activation)

        # Initialize weights and biases
        self.apply(init_linear_weights_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.squeeze()

        mu = self.mu_fcs(x)
        sigma = self.sigma_fcs(x) + 1e-5

        return Normal(mu, sigma)
