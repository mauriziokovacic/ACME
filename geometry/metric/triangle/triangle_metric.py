import torch
from ....math.dot     import *
from ....math.cross   import *
from ....math.norm    import *
from ....math.normvec import *
from ..metric         import *


class TriangleMetric(Metric):
    def __init__(self, *args, **kwargs):
        super(TriangleMetric, self).__init__(*args, **kwargs)

    def points(self, P, T):
        return P[T]

    def edges(self, P, T):
        P0, P1, P2 = self.points(P, T)
        return P2 - P1, P0 - P2, P1 - P0

    def normal(self, P, T):
        L0, L1, L2 = self.edges(P, T)
        return normr(cross(L2, -L1, dim=1))

    def edge_lengths(self, P, T):
        L0, L1, L2 = self.edges(P, T)
        return norm(L0, dim=1), norm(L1, dim=1), norm(L2, dim=1)

    def min_edge_length(self, P, T):
        L = torch.cat(self.edge_lengths(P, T), dim=1)
        return torch.min(L, dim=1)

    def max_edge_length(self, P, T):
        L = torch.cat(self.edge_lengths(P, T), dim=1)
        return torch.max(L, dim=1)

    def area(self, P, T):
        L0, L1, L2 = self.edges(P, T)
        return 0.5 * norm(cross(L0, L1, dim=1), dim=1)

    def inradius(self, P, T):
        nL0, nL1, nL2 = self.edge_lengths(P, T)
        A = self.area(P, T)
        return (2 * A) / (nL0 + nL1 + nL2)

    def circumradius(self, P, T):
        nL0, nL1, nL2 = self.edge_lengths(P, T)
        A = self.area(P, T)
        return (nL0 * nL1 * nL2) / (4 * A)