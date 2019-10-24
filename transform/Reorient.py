from ..geometry.pca import *
from .Transform     import *


class Reorient(Transform):
    def __init__(self):
        super(Reorient, self).__init__()

    def __eval__(self, x, *args, **kwargs):
        C, T = pca(x.pos, dim=0)
        x.pos -= C
        x.pos = torch.matmul(x.pos, T)
        if x.norm:
            x.norm = torch.matmul(x.norm, T)
