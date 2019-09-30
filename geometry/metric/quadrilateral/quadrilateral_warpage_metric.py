from .quadrilateral_metric import *


class QuadrilateralWarpageMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralWarpageMetric, self).__init__(
            name='Quadrilateral Warpage',
            dimension='1',
            acceptable_range=Range(min=0, max=0.7),
            normal_range=Range(min=0, max=2),
            full_range=Range(min=0, max=Inf),
            q_for_unit=0,
        )

    def eval(self, P, T):
        n0, n1, n2, n3 = self.normalized_corner_normals(P, T)
        return 1 - torch.min(torch.pow(dot(n0, n2, dim=1), 3),
                             torch.pow(dot(n1, n3, dim=1), 3), keepdim=True)

