import torch
from ..utility.row         import *
from ..utility.col         import *
from ..utility.LongTensor  import *
from ..utility.FloatTensor import *
from ..utility.indices     import *
from ..math.cross          import *
from ..math.diag           import *
from ..math.norm           import *
from ..topology.adjacency  import *
from ..topology.ispoly     import *


def _gradient(P, T, N, A):
    """
    Creates the gradient operator, starting from the point set P, the topology tensor T, the normal tensor N and the
    triangle area tensor A

    Parameters
    ----------
    P : Tensor
        the (N,3,) point set tensor
    T : LongTensor
        the (3,M,) topology tensor
    N : Tensor
        the (M,3,) triangle normal tensor
    A : Tensor
        the (M,) triangle area tensor

    Returns
    -------
    list
        the gradient operator data
    """

    device = P.device

    def V(i):
        return P[T[i], :]

    n = row(P)
    m = col(T)
    i = LongTensor([], device=device)
    j = LongTensor([], device=device)
    w = FloatTensor([], device=device)
    f = indices(0, m - 1, device=device).squeeze()
    for k in range(row(T)):
        # opposite edge e_i indexes
        s = (k+1) % 3
        t = (k+2) % 3
        # vector N_f^e_i
        wk = cross(V(t) - V(s), N, 1)
        # update the index listing
        i = torch.cat((i, f), dim=0)
        j = torch.cat((j, T[k]), dim=0)
        w = torch.cat((w, wk), dim=0)
    a = diag(torch.reciprocal(A), rows=m)
    e = torch.cat((i.unsqueeze(0), j.unsqueeze(0)), dim=0)
    G = []
    for k in range(col(P)):
        G += [torch.matmul(a, adjacency(e, w[:, k], size=[m, n]))]
    return G


def _divergence(G, A):
    """
    Creates the divergence operator, starting from the gradient operator and the triangles area

    Parameters
    ----------
    G : list
        the gradient operator
    A : Tensor
        the (M,) triangle area tensor

    Returns
    -------
    list
        the divergence operator data
    """

    a = diag(A)
    return [torch.matmul(torch.t(G[0]), a),
            torch.matmul(torch.t(G[1]), a),
            torch.matmul(torch.t(G[2]), a)]


def _laplacian(G, D):
    """
    Creates the laplacian matrix, starting from the gradient and the divergence operators

    Parameters
    ----------
    G : list
        the gradient operator
    D : list
        the divergence operator

    Returns
    -------
    Tensor
        the laplacian matrix
    """

    return torch.matmul(D[0], G[0]) + torch.matmul(D[1], G[1]) + torch.matmul(D[2], G[2])


def differential_operator(T, P):
    """
    Creates the three main differential operators from the given input mesh

    Parameters
    ----------
    T : LongTensor
        the (3,M,) topology tensor
    P : Tensor
        the (N,3,) point set tensor

    Returns
    -------
    (callable, callable, Tensor)
        the gradient operator, the divergence operator, and the laplacian matrix
    """

    assert istri(T), 'Differential operators are defined only on triangle meshes'
    assert col(P) == 3, 'Differential operators are defined only on 3D meshes'
    Pi, Pj, Pk = P[T]
    n = row(P)
    m = col(T)

    N = cross(Pj - Pi, Pk - Pi, 1)
    A = norm(N, 1)
    N = N / A

    G = _gradient(P, T, N, A)
    D = _divergence(G, A)

    def Nabla(u):
        return torch.cat((torch.matmul(G[0], u),
                          torch.matmul(G[1], u),
                          torch.matmul(G[2], u)), dim=1)

    def Div(g):
        return torch.matmul(D[0], g[:, 0]) + \
               torch.matmul(D[1], g[:, 1]) + \
               torch.matmul(D[2], g[:, 2])

    Delta = _laplacian(G, D)

    return Nabla, Div, Delta