from .triangle_metric                       import *
from .triangle_relative_size_squared_metric import *
from .triangle_shape_metric                 import *


class TriangleShapeAndSizeMetric(TriangleMetric):
    def __init__(self):
        super(TriangleShapeAndSizeMetric, self).__init__(
            name='Triangle Shape and Size',
            dimension='1',
            acceptable_range=Range(min=0.25, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=0, max=1),
            q_for_unit='Depends on mean area',
        )

    def eval(self, P, T):
        R = TriangleRelativeSizeSquaredMetric().eval(P, T)
        S = TriangleShapeMetric().eval(P, T)
        return R * S
