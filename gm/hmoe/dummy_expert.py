import torch
from torch import nn


class DummyExpert(nn.Module):
    '''Dummy expert (either identity or zero with residual)'''

    def __init__(self, mode: str = 'identity'):
        super().__init__()
        self.mode = mode

    def forward(self, x, *args, **kwargs):
        if self.mode == 'identity':
            return x
        elif self.mode == 'zero':
            return torch.zeros_like(x)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
