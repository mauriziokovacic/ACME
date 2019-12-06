from ..geometry.vertex_normal import *
from .Transform import *


class VertexNormal(Transform):
    def __init__(self):
        super(VertexNormal, self).__init__()

    def __eval__(self, x, *args, **kwargs):
        if hasattr(x, 'face'):
            x.norm = vertex_normal(x.pos, x.face)
