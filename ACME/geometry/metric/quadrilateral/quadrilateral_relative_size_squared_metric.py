from .quadrilateral_metric      import *
from .quadrilateral_area_metric import *


class QuadrilateralRelativeSizeSquaredMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralRelativeSizeSquaredMetric, self).__init__(
            name='Quadrilateral Relative Size Squared',
            dimension='1',
            acceptable_range=Range(min=0.3, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=0, max=1),
            q_for_unit='Depends on mean area',
        )

    def eval(self, P, T):
        A     = QuadrilateralAreaMetric().eval(P, T)
        A_bar = torch.mean(A)
        R     = A / A_bar
        return torch.pow(torch.min(R, torch.where(R == 0, torch.zeros_like(R), torch.reciprocal(R)))[0], 2)
