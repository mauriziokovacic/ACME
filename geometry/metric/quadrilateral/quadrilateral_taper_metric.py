from .quadrilateral_metric import *


class QuadrilateralTaperMetric(QuadrilateralMetric):
    def __init__(self):
        super(QuadrilateralTaperMetric, self).__init__(
            name='Quadrilateral Taper',
            dimension='1',
            acceptable_range=Range(min=0.25, max=1),
            normal_range=Range(min=0, max=1),
            full_range=Range(min=0, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        X12 = self.cross_derivatie_lengths(P, T)[0]
        X   = torch.cat(self.principal_axes_lengths(P, T), dim=1)
        return X12 / torch.min(X, dim=1, keepdim=True)
