from typing import Callable

import torch
import torch.nn as nn

from common.models.utils import init_linear_weights_xavier

class CNN(nn.Module):
    """ Convolutional Neural Network """

    def __init__(
        self,
        input_size: list,
        output_size: int,
        conv_layers: list,
        hidden_sizes: list,
        hidden_activation: Callable = nn.ReLU(),
        output_activation: Callable = nn.Sequential(),
    ):
        super().__init__()

        self.cnn = nn.Sequential()
        self.fcs = nn.Sequential()

        # Conv Layers
        for i, conv in enumerate(conv_layers):
            self.cnn.add_module(f"cnn{i}", conv)
            self.cnn.add_module(f"cnn_act{i}", hidden_activation)

        flatten_size = self.cnn(torch.zeros(1, *input_size)).view(-1).size(0)

        # Hidden Layers
        prev_size = flatten_size
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
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x
