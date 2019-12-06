from .quadrilateral_metric import *


class QuadrilateralConditionMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralConditionMetric, self).__init__(
            name='Quadrilateral Condition',
            dimension='1',
            acceptable_range=Range(min=1, max=4),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        L = torch.pow(torch.cat(self.edge_lengths(P, T), dim=1), 2)
        a = torch.cat(self.areas(P, T), dim=1)
        return 0.5 * torch.max(torch.cat(((L[:, 0]+L[:, 3])/a[:, 0],
                                          (L[:, 1]+L[:, 0])/a[:, 1],
                                          (L[:, 2]+L[:, 1])/a[:, 2],
                                          (L[:, 3]+L[:, 2])/a[:, 3]), dim=1), dim=1, keepdim=True)[0]
