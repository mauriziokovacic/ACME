from .quadrilateral_metric import * 


class QuadrilateralScaledJacobianMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralScaledJacobianMetric, self).__init__(
            name='Quadrilateral Scaled Jacobian',
            dimension='1',
            acceptable_range=Range(min=0.3, max=1),
            normal_range=Range(min=-1, max=1),
            full_range=Range(min=-1, max=1),
            q_for_unit=1,
        )

    def eval(self, P, T):
        L = torch.cat(self.edge_lengths(P, T), dim=1)
        a = torch.cat(self.areas(P, T), dim=1)
        return torch.min(torch.cat((a[:, 0] / (L[:, 0] * L[:, 3]),
                                    a[:, 1] / (L[:, 1] * L[:, 0]),
                                    a[:, 2] / (L[:, 2] * L[:, 1]),
                                    a[:, 3] / (L[:, 3] * L[:, 2]),
                                    ), dim=1),
                         dim=1, keepdim=True)[0]
