from ..math.randrotm    import *
from .Transform         import *


class RandomRotation(Transform):
    def __init__(self, attr=['pos', 'norm'], name='RandomRotation'):
        super(RandomRotation, self).__init__(name=name)
        self.attr = attr if islist(attr) else [attr]

    def __eval__(self, x, *args, **kwargs):
        T = randrotm(1, x.pos.device)
        for attr in self.attr:
            if hasattr(x, attr):
                d = getattr(x, attr)
                if d:
                    setattr(x, attr, torch.matmul(d, T))
