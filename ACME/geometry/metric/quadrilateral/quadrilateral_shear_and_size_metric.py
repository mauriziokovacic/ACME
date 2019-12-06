from .quadrilateral_metric                       import *
from .quadrilateral_relative_size_squared_metric import *
from .quadrilateral_shear_metric                 import *


class QuadrilateralShearAndSizeMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralShearAndSizeMetric, self).__init__(
            name='Quadrilateral Shape And Size',
            dimension='1',
            acceptable_range=Range(min=0.2, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=0, max=1),
            q_for_unit='Depends on mean area',
        )

    def eval(self, P, T):
        R = QuadrilateralRelativeSizeSquaredMetric().eval(P, T)
        H = QuadrilateralShearMetric().eval(P, T)
        return R*H