from ..geometry.barycenter import *
from .Transform            import *


class FaceBarycenter(Transform):
    def __init__(self, attr='fpos'):
        super(FaceBarycenter, self).__init__()
        self.attr = attr

    def __eval__(self, x, *args, **kwargs):
        setattr(x, self.attr, barycenter(x.pos, T=x.face))

    def extra_repr(self):
        return 'attr={}'.format(self.attr)

