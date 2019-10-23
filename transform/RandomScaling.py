import torch
from ..utility.islist import *
from ..math.normvec   import *
from .Transform       import *


class RandomScaling(Transform):
    def __init__(self, attr=['pos', 'norm'], name='RandomScaling'):
        super(RandomScaling, self).__init__(name=name)
        self.attr = attr if islist(attr) else [attr]

    def __eval__(self, x, *args, **kwargs):
        T = torch.diag(torch.rand(3, x.pos.device) * 2 - 1)
        for attr in self.attr:
            if hasattr(x, attr):
                d = getattr(x, attr)
                if d:
                    if attr == 'norm':
                        setattr(x, attr, normr(torch.matmul(d, T)))
                    else:
                        setattr(x, attr, torch.matmul(d, T))
