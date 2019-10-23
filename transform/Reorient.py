from ..geometry.pca import *
from .Transform     import *


class Reorient(Transform):
    def __init__(self, name='Reorient'):
        super(Reorient, self).__init__(name=name)

    def __eval__(self, x, *args, **kwargs):
        C, T = pca(x.pos, dim=0)
        x.pos -= C
        x.pos = torch.matmul(x.pos, T)
        if x.norm:
            x.norm = torch.matmul(x.norm, T)
