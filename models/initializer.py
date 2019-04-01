import torch.nn as nn
# Xavier Initialization


def init_linear_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
