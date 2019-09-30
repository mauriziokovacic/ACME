from .triangle_metric import *


class TriangleAreaMetric(TriangleMetric):
    def __init__(self):
        super(TriangleAreaMetric, self).__init__(
            name='Triangle Area',
            dimension='L^2',
            acceptable_range=Range(min=0, max=Inf),
            normal_range=Range(min=0, max=Inf),
            full_range=Range(min=0, max=Inf),
            q_for_unit=SQRT3/4,
        )

    def eval(self, P, T):
        return self.area(P, T)
