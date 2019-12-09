from .triangle_metric import *


class TriangleRadiusRatioMetric(TriangleMetric):
    def __init__(self):
        super(TriangleRadiusRatioMetric, self).__init__(
            name='Triangle Radius Ratio',
            dimension='1',
            acceptable_range=Range(min=1, max=3),
            normal_range=Range(min=-1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        R = self.circumradius(P, T)
        r = self.inradius(P, T)
        return R / (2 * r)
