import torch
from .Transform import *


class Normalize(Transform):
    def __init__(self):
        super(Normalize, self).__init__()

    def __eval__(self, x, *args, **kwargs):
        min = torch.min(x.pos, dim=0)[0]
        max = torch.max(x.pos, dim=0)[0]
        d = torch.max(max - min)[0]
        c = 0.5 * (max + min)
        x.pos -= c
        x.pos *= 2/d
