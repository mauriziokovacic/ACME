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
        L0, L1, L2 = self.edges(P, T)
        A          = self.area(P, T)
        n          = self.normal(P, T)
        J = torch.cat((L2.unsqueeze(2), -L1.unsqueeze(2), n.unsqueeze(2)), dim=2).det().unsqueeze(1)
        return (J * SQRT3) / A
