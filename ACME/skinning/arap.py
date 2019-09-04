from ACME.math import *
from ACME.geometry import *

def arap_energy(P, P_prime, R, wij, E):
    Pi_prime, Pj_prime = P_prime[E]
    Pi, Pj = P[E]
    Ri = R[E[0]]
    return torch.sum(wij * sqnorm((Pi_prime - Pj_prime) - torch.matmul(Ri, torch.t(Pi - Pj).unsqueeze(2).permute(1, 0, 2)).squeeze(), 1))


def arap_S(Eij, Dij, Eij_prime):
    return torch.matmul(torch.t(Eij), torch.matmul(Dij, Eij_prime))


def arap_R(Si):
    U, _, V = torch.svd(Si)
    return torch.matmul(V, torch.t(U))


def arap_Si(P, P_prime, A, i):
    j         = find(A[i, :])
    Eij       = P[i, :] - P[j, :]
    Eij_prime = P_prime[i, :] - P_prime[j, :]
    Dij       = diag(A[i, j])
    return arap_S(Eij, Dij, Eij_prime)


def arap_metric(P, P_prime, A, E):
    R = torch.cat([arap_R(arap_Si(P, P_prime, A, i)).unsqueeze(0) for i in range(row(P))], dim=0)
    return arap_energy(P, P_prime, R, A[tuple(E)], E)