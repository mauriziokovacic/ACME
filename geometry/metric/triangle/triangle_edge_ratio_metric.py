from .triangle_metric import *


class TriangleEdgeRatioMetric(TriangleMetric):
    def __init__(self):
        super(TriangleEdgeRatioMetric, self).__init__(
            name='Triangle Edge Ratio',
            dimension='1',
            acceptable_range=Range(min=1, max=1.3),
            normal_range=Range(min=1, max=Inf),
            full_range=Range(min=1, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        L = torch.cat(self.edge_lengths(P, T), dim=1)
        return torch.max(L, dim=1, keepdim=True)[0] / torch.min(L, dim=1, keepdim=True)[0]
