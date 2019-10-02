from ..geometry.triangle_normal import *
from .Transform                 import *


class FaceNormal(Transform):
    def __init__(self, name='FaceNormal'):
        super(FaceNormal, self).__init__(name=name)

    def __eval__(self, x, *args, **kwargs):
        setattr(x, 'fnorm', triangle_normal(x.pos, x.face))

