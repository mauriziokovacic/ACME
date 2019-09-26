from .quadrilateral_metric import *


class QuadrilateralAreaMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralAreaMetric, self).__init__(
            name='Quadrilateral Area',
            dimension='L^2',
            acceptable_range=Range(min=0, max=Inf),
            normal_range=Range(min=0, max=Inf),
            full_range=Range(min=-Inf, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        return 1/4 * torch.sum(torch.cat(self.areas(P, T), dim=1), dim=1, keepdim=True)
