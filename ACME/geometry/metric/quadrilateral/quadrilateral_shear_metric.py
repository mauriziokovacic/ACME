from .quadrilateral_metric                 import *
from .quadrilateral_scaled_jacobian_metric import *


class QuadrilateralShearMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralShearMetric, self).__init__(
            name='Quadrilateral Shear',
            dimension='1',
            acceptable_range=Range(min=0.3, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=0, max=1),
            q_for_unit=1,
        )

    def eval(self, P, T):
        return QuadrilateralScaledJacobianMetric().eval(P, T)

