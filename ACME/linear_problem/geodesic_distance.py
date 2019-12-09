from ..math.normvec                    import *
from ..geometry.area                   import *
from ..geometry.poly_edges             import *
from ..geometry.differential_operators import *
from .heat_diffusion                   import *
from .kronecker_delta                  import *


def geodesic_distance(P, T, i, eps=0.0001):
    """
    Computes the geodesic distance of points i

    Parameters
    ----------
    P : Tensor
        the (N,3,) points set tensor
    T : LongTensor
        the (3,M,) topology tensor
    i : LongTensor
        the (I,) indices tensor
    eps : float (optional)
        a regularizer value (default is 0.0001)

    Returns
    -------
    Tensor
        the (N,) geodesic distance tensor
    """

    A       = spdiag(barycentric_area(P, T))
    t       = poly_edges_mean_length(P, T)**2
    k       = kronecker_delta(P.size(0), i, device=P.device)
    G, D, L = differential_operator(T, P)
    u       = heat_diffusion(A, t, L, k, eps=eps)
    du      = normr(G(u))
    d       = D(G)
    return poisson_equation(L, d, eps=eps)
