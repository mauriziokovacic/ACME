from ..math.randrotm    import *
from .Transform         import *


class RandomRotation(Transform):
    def __init__(self, attr=['pos', 'norm']):
        super(RandomRotation, self).__init__()
        self.attr = attr if islist(attr) else [attr]

    def __eval__(self, x, *args, **kwargs):
        T = randrotm(1, x.pos.device)
        for attr in self.attr:
            if hasattr(x, attr):
                d = getattr(x, attr)
                if d:
                    setattr(x, attr, torch.matmul(d, T))

    def extra_repr(self):
        return 'attr={}'.format(self.attr if len(self.attr) > 1 else self.attr[0])
