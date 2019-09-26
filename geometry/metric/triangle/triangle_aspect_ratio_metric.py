from .triangle_metric import *


class TriangleAspectRatioMetric(TriangleMetric):
    def __init__(self):
        super(TriangleAspectRatioMetric, self).__init__(
            name='Triangle Aspect Ratio',
            dimension='1',
            acceptable_range=Range(min=1, max=1.3),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        Lmax = self.max_edge_length(P, T)[0]
        r    = self.inradius(P, T)
        return Lmax / (2*SQRT3*r)
