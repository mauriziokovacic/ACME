from ..geometry.area import *
from .Transform      import *


class FaceArea(Transform):
    def __init__(self, attr='farea'):
        super(FaceArea, self).__init__()
        self.attr = attr

    def __eval__(self, x, *args, **kwargs):
        setattr(x, self.attr, triangle_area(x.pos, T=x.face))

    def __extra_repr__(self):
        return 'attr={}'.format(self.attr)


