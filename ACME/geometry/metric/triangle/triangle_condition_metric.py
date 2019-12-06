from .triangle_metric import *


class TriangleConditionMetric(TriangleMetric):
    def __init__(self):
        super(TriangleConditionMetric, self).__init__(
            name='Triangle Condition',
            dimension='1',
            acceptable_range=Range(min=1, max=1.3),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        L0, L1 = self.edges(P, T)[:2]
        A      = self.area(P, T)
        return (dot(L1, L1, dim=1) + dot(L0, L0, dim=1) + dot(L0, L1, dim=1)) / (2*SQRT3*A)
