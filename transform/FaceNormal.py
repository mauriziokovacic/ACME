from ..geometry.triangle_normal import *
from .Transform                 import *


class FaceNormal(Transform):
    def __init__(self, attr='fnorm', name='FaceNormal'):
        super(FaceNormal, self).__init__(name=name)
        self.attr = attr

    def __eval__(self, x, *args, **kwargs):
        setattr(x, self.attr, triangle_normal(x.pos, x.face))

