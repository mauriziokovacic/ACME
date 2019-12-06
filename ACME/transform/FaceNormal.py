from ..geometry.triangle_normal import *
from .Transform                 import *


class FaceNormal(Transform):
    def __init__(self, attr='fnorm'):
        super(FaceNormal, self).__init__()
        self.attr = attr

    def __eval__(self, x, *args, **kwargs):
        setattr(x, self.attr, triangle_normal(x.pos, x.face))

    def extra_repr(self):
        return 'attr={}'.format(self.attr)
