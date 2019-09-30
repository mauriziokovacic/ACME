from .quadrilateral_metric import *


class QuadrilateralJacobianMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralJacobianMetric, self).__init__(
            name='Quadrilateral Jacobian',
            dimension='L^2',
            acceptable_range=Range(min=0, max=Inf),
            normal_range=Range(min=0, max=Inf),
            full_range=Range(min=-Inf, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        return torch.min(torch.cat(self.areas(P, T), dim=1), dim=1, keepdim=True)[0]
