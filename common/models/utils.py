import torch.nn as nn

def init_linear_weights_xavier(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

class NoisyLinear(nn.Module):
    def __init__(self, input_size, output_size, std_init=0.4):
        super().__init__()

        self.weight_mu = nn.Parameter(torch.FloatTensor(output_size, input_size))
        self.weight_sigma = nn.Parameter()