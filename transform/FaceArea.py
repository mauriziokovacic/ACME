from ..geometry.area import *
from .Transform      import *


class FaceArea(Transform):
    def __init__(self, attr='farea', name='FaceArea'):
        super(FaceArea, self).__init__(name=name)
        self.attr = attr

    def __eval__(self, x, *args, **kwargs):
        setattr(x, self.attr, triangle_area(x.pos, T=x.face))
