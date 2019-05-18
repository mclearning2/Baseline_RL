

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
            self.fcs.add_module(f"output", nn.Linear(prev_size, output_size))
            self.fcs.add_module(f"output_act", output_activation)

        # Initialized
        self.apply(init_linear_weights_xavier)

    def forward(self, x: torch.Tensor):
        """Forward method implementation"""       

        return self.fcs(x)

class SepMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes1: list,
        hidden_sizes2: list,
        output_size1: int,
        output_size2: int,
        hidden_activation: Callable = nn.ReLU(),
        output_activation1: Callable = nn.Sequential(), # identity
        output_activation2: Callable = nn.Sequential(), # identity
    ):
        self.mlp1 = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes1,
            output_size=output_size1,
            hidden_activation=hidden_activation,
            output_activation=output_activation1
        )
        
        self.mlp2 = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes2,
            output_size=output_size2,
            hidden_activation=hidden_activation,
            output_activation=output_activation2
        )

        self.apply(init_linear_weights_xavier)

    def forward(self, x):
        output1 = self.mlp1(x)
        output2 = self.mlp2(x)

        return output1, output2

class ShareMLP(MLP):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_sizes1: list,
        output_sizes2: list,
        hidden_activation: Callable = nn.ReLU(),
        output_activation1: Callable = nn.Sequential(), # identity
        output_activation2: Callable = nn.Sequential(), # identity
    ):
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=0,
            hidden_activation=hidden_activation,            
        )
        
        self.mlp1 = nn.Sequential()
        self.mlp2 = nn.Sequential()

        last_size = hidden_sizes[-1] if hidden_sizes else input_size

        prev_size = last_size
        for i in range(len(output_sizes1)):
            next_size = output_sizes1[i]
            self.mlp1.add_module(f"hidden1_fc{i}", nn.Linear(prev_size, next_size))
            if i < len(output_sizes1) - 1:
                self.mlp1.add_module(f"hidden1_fc_act{i}", hidden_activation)
            else:
                self.mlp1.add_module(f"hidden1_fc_act{i}", output_activation1)
            prev_size = next_size

        prev_size = last_size
        for i in range(len(output_sizes2)):
            next_size = output_sizes2[i]
            self.mlp2.add_module(f"hidden2_fc{i}", nn.Linear(prev_size, next_size))
            if i < len(output_sizes2) - 1:
                self.mlp2.add_module(f"hidden2_fc_act{i}", hidden_activation)
            else:
                self.mlp2.add_module(f"hidden2_fc_act{i}", output_activation2)
            prev_size = next_size

        self.apply(init_linear_weights_xavier)

    def forward(self, x):
        hidden = super().forward(x)

        output1 = self.mlp1(hidden)
        output2 = self.mlp2(hidden)

        return output1, output2

class CategoricalDist(MLP):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int = 0,
        hidden_activation: Callable = nn.ReLU(),
        output_activation: Callable = nn.Softmax(),
    ):
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )

        self.apply(init_linear_weights_xavier)

    def forward(self, x):
        probs = super().forward(x)
        return Categorical(probs)

class NormalDist(MLP):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int = 0,
        hidden_activation: Callable = nn.ReLU(),
        output_activation: Callable = nn.Tanh(),
        std:float = 0.0,
    ):
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
        )
        self.std = std

        self.apply(init_linear_weights_xavier)

    def forward(self, x):
        mu = super().forward(x)
        log_std = (torch.ones_like(mu) * self.std)
        return Normal(mu, log_std.exp())
    
class SepNormalDist(SepMLP):
    def __init__(
        self,
        input_size: int,
        mu_hidden_sizes: list,
        sigma_hidden_sizes: list,
        mu_output_size: int,
        sigma_output_size: int,
        hidden_activation: Callable = nn.ReLU(),
        mu_output_activation: Callable = nn.Sequential(), # identity
        sigma_output_activation: Callable = nn.Sequential(), # identity
    ):
        super().__init__(
            input_size=input_size,
            hidden_sizes1=mu_hidden_sizes,
            hidden_sizes2=sigma_hidden_sizes,
            output_size1=mu_output_size,
            output_size2=sigma_output_size,
            hidden_activation=hidden_activation,
            output_activation1=mu_output_activation,
            output_activation2=sigma_output_activation
        )

    def forward(self, x):
        mu, sigma = super().forward(x)
        
        return Normal(mu, sigma)

class ShareNormalDist(ShareMLP):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        mu_output_size: int,
        sigma_output_size: int,
        hidden_activation: Callable = nn.ReLU(),
        mu_output_activation: Callable = nn.Sequential(), # identity
        sigma_output_activation: Callable = nn.Sequential(), # identity
    ):
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size1=mu_output_size,
            output_size2=sigma_output_size,
            hidden_activation=hidden_activation,
            output_activation1=mu_output_activation,
            output_activation2=sigma_output_activation
        )

    def forward(self, x):
        mu, sigma = super().forward(x)
        
        return Normal(mu, sigma)

class SepActorCritic(nn.Module):
    def __init__(
        self,
        actor: Union[MLP, CategoricalDist, NormalDist, SepNormalDist, ShareNormalDist],
        critic: MLP,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic

        self.apply(init_linear_weights_xavier)

    def forward(self, x):
        dist = self.actor(x)
        value = self.critic(x)

        return dist, value

class ShareActorCritic(ShareMLP):
    def __init__(
        self,
        input_size:int,
        hidden_sizes: list,
        actor_output_size: int,
        critic_output_size: int,
        dist: Union[Normal, Categorical],
        std: float = 0.0,
        hidden_activation: Callable = nn.ReLU(),
        actor_output_activation: Callable = nn.Sequential(), # identity
        critic_output_activation: Callable = nn.Sequential(), # identity
    ):
        super().__init__(
            self,
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size1=actor_output_size,
            output_size2=critic_output_size,
            hidden_activation=hidden_activation,
            output_activation1=actor_output_activation,
            output_activation2=critic_output_activation
        )
        self.dist = dist
        self.std = std

        self.apply(init_linear_weights_xavier)

    def forward(self, x):

        actor, critic = super().forward(x)

        if self.dist == Normal:
            log_std = (torch.ones_like(ac) * self.std)

            return self.dist(actor, log_std.exp()), critic
        elif self.dist == Categorical:
            return self.dist(actor), critic
        
        else:
            raise TypeError("Normal or Categorical")
