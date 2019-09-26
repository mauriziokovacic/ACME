from .triangle_metric import *


class TriangleScaledJacobianMetric(TriangleMetric):
    def __init__(self):
        super(TriangleScaledJacobianMetric, self).__init__(
            name='Triangle Scaled Jacobian',
            dimension='1',
            acceptable_range=Range(min=0.5, max=(2*SQRT3)/3),
            normal_range=Range(min=-(2*SQRT3)/3, max=(2*SQRT3)/3),
            full_range=Range(min=-Inf, max=Inf),
            q_for_unit=1,
        )

    def eval(self, P, T):
        L0, L1, L2 = self.edges(P, T)
        Lmax = torch.max(norm(L0, dim=1) * norm(L1, dim=1),
                         norm(L0, dim=1) * norm(L2, dim=1), keepdim=True)[0]
        Lmax = torch.max(Lmax,
                         norm(L1, dim=1) * norm(L2, dim=1), keepdim=True)[0]
        n = self.normal(P, T)
        J = torch.det()
        if dot(n, cross(L2, L1, dim=1), dim=1):
            J = -J
        return (2*SQRT3)/3 * (J / Lmax)
