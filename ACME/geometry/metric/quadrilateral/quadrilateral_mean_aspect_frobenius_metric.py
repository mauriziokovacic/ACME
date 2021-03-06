from .quadrilateral_metric                       import *
from ..triangle.triangle_aspect_frobenius_metric import *


class QuadrilateralMeanAspectFrobeniusMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralMeanAspectFrobeniusMetric, self).__init__(
            name='Quadrilateral Mean Aspect Frobenius',
            dimension='1',
            acceptable_range=Range(min=1, max=1.3),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        F = TriangleAspectFrobeniusMetric()
        return (1/4) * torch.sum(torch.cat((F(P, T[(3, 0, 1), :]),
                                            F(P, T[(0, 1, 2), :]),
                                            F(P, T[(1, 2, 3), :]),
                                            F(P, T[(2, 3, 0), :])), dim=1), dim=1, keepdim=True)
