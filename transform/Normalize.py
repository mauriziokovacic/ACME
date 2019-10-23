import torch
from .Transform import *


class Normalize(Transform):
    def __init__(self, name='Rescale'):
        super(Normalize, self).__init__(name=name)

    def __eval__(self, x, *args, **kwargs):
        min = torch.min(x.pos, dim=0)
        max = torch.max(x.pos, dim=0)
        d = torch.max(max - min)
        c = 0.5 * (max + min)
        x.pos -= c
        x.pos *= 2/d
