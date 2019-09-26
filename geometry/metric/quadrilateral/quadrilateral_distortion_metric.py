from .quadrilateral_metric      import *
from .quadrilateral_area_metric import *


class QuadrilateralDistortionMetric(Metric):
    def __init__(self):
        super(QuadrilateralDistortionMetric, self).__init__(
            name='Quadrilateral Distortion',
            dimension='1',
            acceptable_range=Range(min=0.5, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=-Inf, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        A = QuadrilateralAreaMetric().eval(P, T)

        return (4 * torch.det(J)) / A