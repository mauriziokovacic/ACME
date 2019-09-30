from .quadrilateral_metric      import *
from .quadrilateral_area_metric import *


class QuadrilateralAspectRatioMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralAspectRatioMetric, self).__init__(
            name='Quadrilateral Aspect Ratio',
            dimension='1',
            acceptable_range=Range(min=1, max=1.3),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        L = torch.cat(self.edge_lengths(P, T), dim=1)
        A = QuadrilateralAreaMetric()(P, T)
        return (torch.max(L, dim=1, keepdim=True) * torch.sum(L, dim=1, keepdim=True)) / (4 * A)
