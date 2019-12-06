import torch
from .init import *


class FixedGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, matrix, bias=False):
        super(FixedGraphConv, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, in_channels, out_channels))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, out_channels))
        else:
            self.register_parameter('bias', None)
        self.matrix = matrix
        self.reset_parameters()

    def reset_parameters(self):
        init(self.weight, 'xavier')
        init(self.bias, 'zero')

    def forward(self, x):
        b = 0 if self.bias is None else self.bias
        return (torch.matmul(self.matrix, torch.matmul(x, self.weight)) + b).squeeze()

    def extra_repr(self):
        return '{}, {}, bias={}'.format(self.weight.size(-2), self.weight.size(-1), self.bias is not None)
