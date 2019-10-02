from ..geometry.barycenter import *
from .Transform            import *


class FaceBarycenter(Transform):
    def __init__(self, name='FaceBarycenter'):
        super(FaceBarycenter, self).__init__(name=name)

    def eval(self, x, *args, **kwargs):
        setattr(x, 'fpos', barycenter(x.pos, T=x.face))

