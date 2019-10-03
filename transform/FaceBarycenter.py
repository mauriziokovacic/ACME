from ..geometry.barycenter import *
from .Transform            import *


class FaceBarycenter(Transform):
    def __init__(self, attr='fpos', name='FaceBarycenter'):
        super(FaceBarycenter, self).__init__(name=name)
        self.attr = 'fpos'

    def eval(self, x, *args, **kwargs):
        setattr(x, self.attr, barycenter(x.pos, T=x.face))

