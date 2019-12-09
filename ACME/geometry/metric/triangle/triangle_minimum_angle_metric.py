from ....math.angle   import *
from ....math.rad2deg import *
from .triangle_metric import *


class TriangleMinimumAngleMetric(TriangleMetric):
    def __init__(self):
        super(TriangleMinimumAngleMetric, self).__init__(
            name='Triangle Minimum Angle',
            dimension='A^1',
            acceptable_range=Range(min=30.0, max=60.0),
            normal_range=Range(min=0.0, max=60.0),
            full_range=Range(min=0.0, max=360.0),
            q_for_unit=60.0,
        )

    def eval(self, P, T):
        L0, L1, L2 = self.edges(P, T)
        q = torch.min(angle(normr(L0), normr(L1), dim=1),
                      angle(normr(L1), normr(L2), dim=1), keepdim=True)[0]
        q = torch.min(q, angle(normr(L2), normr(L0), dim=1))[0]
        return rad2deg(q)
