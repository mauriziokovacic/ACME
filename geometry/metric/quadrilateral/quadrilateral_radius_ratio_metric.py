from .quadrilateral_metric import *


class QuadrilateralRadiusRatioMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralRadiusRatioMetric, self).__init__(
            name='Quadrilateral Radius Ratio',
            dimension='1',
            acceptable_range=Range(min=1, max=1.3),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        L    = torch.cat(self.edge_lengths(P, T), dim=1)
        a    = torch.cat(self.areas(P, T), dim=1)
        hmax = torch.max(torch.max(L, dim=1, keepdim=True)[0], self.max_diagonal_length(P, T), keepdim=True)
        L2   = torch.sum(torch.pow(L, 2), dim=1, keepdim=True)
        A    = torch.abs(a / 2)
        return (L2*hmax)/torch.min(A, dim=1, keepdim=True)[0]
