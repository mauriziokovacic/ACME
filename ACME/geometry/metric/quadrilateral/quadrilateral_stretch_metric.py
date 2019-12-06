from .quadrilateral_metric import *


class QuadrilateralStretchMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralStretchMetric, self).__init__(
            name='Quadrilateral Stretch',
            dimension='1',
            acceptable_range=Range(min=0.25, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=0, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        L    = torch.cat(self.edge_lengths(P, T), dim=1)
        Dmax = self.max_diagonal_length(P, T)
        return SQRT2 * torch.min(L, dim=1, keepdim=True)[0] / Dmax

