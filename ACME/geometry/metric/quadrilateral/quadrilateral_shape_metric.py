from .quadrilateral_metric import * 


class QuadrilateralShapeMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralShapeMetric, self).__init__(
            name='Quadrilateral Shape',
            dimension='1',
            acceptable_range=Range(min=0.3, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=0, max=1),
            q_for_unit=1,
        )

    def eval(self, P, T):
        L = torch.pow(torch.cat(self.edge_lengths(P, T), dim=1), 2)
        a = torch.cat(self.areas(P, T), dim=1)
        return 2*torch.min(torch.cat((a[:, 0] / (L[:, 0] + L[:, 3]),
                                      a[:, 1] / (L[:, 1] + L[:, 0]),
                                      a[:, 2] / (L[:, 2] + L[:, 1]),
                                      a[:, 3] / (L[:, 3] + L[:, 2]),
                                     ), dim=1),
                           dim=1, keepdim=True)[0]
