from ..geometry.area import *
from .Transform      import *


class FaceArea(Transform):
    def __init__(self, name='FaceArea'):
        super(FaceArea, self).__init__(name=name)

    def __eval__(self, x, *args, **kwargs):
        setattr(x, 'farea', triangle_area(x.pos, T=x.face))
