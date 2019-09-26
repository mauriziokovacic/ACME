from .triangle_metric import *


class TriangleAspectFrobeniusMetric(TriangleMetric):
    def __init__(self):
        super(TriangleAspectFrobeniusMetric, self).__init__(
            name='Triangle Aspect Frobenius',
            dimension='1',
            acceptable_range=Range(min=1, max=1.3),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        nL0, nL1, nL2 = self.edge_lengths(P, T)
        A             = self.area(P, T)
        return (nL0**2 + nL1**2 + nL2**2) / (4*SQRT3*A)
