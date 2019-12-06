from .triangle_metric           import *
from .triangle_condition_metric import *


class TriangleShapeMetric(TriangleMetric):
    def __init__(self):
        super(TriangleShapeMetric, self).__init__(
            name='Triangle Shape',
            dimension='1',
            acceptable_range=Range(min=0.25, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=0, max=1),
            q_for_unit=1,
        )

    def eval(self, P, T):
        q = TriangleConditionMetric().eval(P, T)
        return torch.reciprocal(q)
