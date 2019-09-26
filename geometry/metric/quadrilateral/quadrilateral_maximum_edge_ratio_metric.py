from .quadrilateral_metric import *


class QuadrilateralMaximumEdgeRatioMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralMaximumEdgeRatioMetric, self).__init__(
            name='Quadrilateral Maximum Edge Ratio',
            dimension='1',
            acceptable_range=Range(min=1, max=1.3),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        X1, X2 = self.principal_axes(P, T)
        q = norm(X1, dim=1) / norm(X2, dim=1)
        return torch.max(q, torch.reciprocal(q), keepdim=True)[0]
