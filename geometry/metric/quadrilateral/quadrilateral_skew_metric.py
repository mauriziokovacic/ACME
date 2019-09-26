from .quadrilateral_metric import *


class QuadrilateralSkewMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralSkewMetric, self).__init__(
            name='Quadrilateral Skew',
            dimension='1',
            acceptable_range=Range(min=0.5, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=0, max=1),
            q_for_unit=1,
        )

    def eval(self, P, T):
        X1, X2 = self.normalized_principal_axes(P, T)
        return torch.abs(dot(X1, X2, dim=1))