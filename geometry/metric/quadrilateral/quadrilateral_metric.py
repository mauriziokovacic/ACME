import torch
from ....math.norm    import *
from ....math.normvec import *
from ....math.cross   import *
from ....math.dot     import *
from ..metric         import *


class QuadrilateralMetric(Metric):
    def __init__(self, *args, **kwargs):
        super(QuadrilateralMetric, self).__init__(*args, **kwargs)

    def points(self, P, T):
        return tuple(P[T])

    def edges(self, P, T):
        P0, P1, P2, P3 = self.points(P, T)
        return P1 - P0, P2 - P1, P3 - P2, P0 - P3

    def edge_lengths(self, P, T):
        L0, L1, L2, L3 = self.edges(P, T)
        return norm(L0, dim=1), norm(L1, dim=1), norm(L2, dim=1), norm(L3, dim=1)

    def min_edge_length(self, P, T):
        L = torch.cat(self.edge_lengths(P, T), dim=1)
        return torch.min(L, dim=1)

    def max_edge_length(self, P, T):
        L = torch.cat(self.edge_lengths(P, T), dim=1)
        return torch.max(L, dim=1)

    def diagonals(self, P, T):
        P0, P1, P2, P3 = self.points(P, T)
        return P2 - P0, P3 - P1

    def diagonal_lengths(self, P, T):
        D0, D1 = self.diagonals(P, T)
        return norm(D0, dim=1), norm(D1, dim=1)

    def min_diagonal_length(self, P, T):
        L = torch.cat(self.diagonal_lengths(P, T), dim=1)
        return torch.min(L, dim=1)

    def max_diagonal_length(self, P, T):
        L = torch.cat(self.diagonal_lengths(P, T), dim=1)
        return torch.max(L, dim=1)

    def principal_axes(self, P, T):
        P0, P1, P2, P3 = self.points(P, T)
        return (P1 - P0) + (P2 - P3), (P2 - P1) + (P3 - P0)

    def principal_axes_lengths(self, P, T):
        X1, X2 = self.principal_axes(P, T)
        return norm(X1, dim=1), norm(X2, dim=1)

    def normalized_principal_axes(self, P, T):
        X1, X2 = self.principal_axes(P, T)
        return normr(X1), normr(X2)

    def cross_derivatives(self, P, T):
        P0, P1, P2, P3 = self.points(P, T)
        return (P0 - P1) + (P2 - P3), (P0 - P3) + (P2 - P1)

    def cross_derivatie_lengths(self, P, T):
        X12, X21 = self.cross_derivatives(P, T)
        return norm(X12, dim=1), norm(X21, dim=1)

    def normalized_cross_derivatives(self, P, T):
        X12, X21 = self.cross_derivatives(P, T)
        return normr(X12), normr(X21)

    def corner_normals(self, P, T):
        L0, L1, L2, L3 = self.edges(P, T)
        return cross(L3, L0, dim=1), cross(L0, L1, dim=1), cross(L1, L2, dim=1), cross(L2, L3, dim=1)

    def normalized_corner_normals(self, P, T):
        N0, N1, N2, N3 = self.corner_normals(P, T)
        return normr(N0), normr(N1), normr(N2), normr(N3)

    def center_normal(self, P, T):
        X1, X2 = self.principal_axes(P, T)
        return cross(X1, X2, dim=1)

    def normalized_center_normal(self, P, T):
        Nc = self.center_normal(P, T)
        return normr(Nc)

    def areas(self, P, T):
        N0, N1, N2, N3 = self.corner_normals(P, T)
        nc = self.normalized_center_normal(P, T)
        return dot(nc, N0, dim=1), dot(nc, N1, dim=1), dot(nc, N2, dim=1), dot(nc, N3, dim=1)

    def is_degenerate(self, P, T):
        return torch.any(torch.cat(self.areas(P, T), dim=1) < 0, dim=1)
