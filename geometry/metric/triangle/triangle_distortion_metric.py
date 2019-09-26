from .triangle_metric import *


class TriangleDistortionMetric(TriangleMetric):
    def __init__(self):
        super(TriangleDistortionMetric, self).__init__(
            name='Triangle Distortion',
            dimension='1',
            acceptable_range=Range(min=0.5, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=-Inf, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        raise NotImplementedError
