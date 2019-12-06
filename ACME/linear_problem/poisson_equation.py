from ..utility.issquare import *
from .linear_problem    import *


def poisson_equation(L, f, eps=0.0001):
    """
    Solves the Poisson equation for the given Laplace matrix

    Parameters
    ----------
    L : Tensor or SparseTensor
        the (N,N,) Laplace matrix
    f : Tensor
        the (N,D,) known values tensor
    eps : float (optional)
        a regularizer value (default is 0.0001)

    Returns
    -------
    Tensor
        the (N,D,) solution tensor
    """

    if not issquare(L):
        raise RuntimeError('Laplace matrix should be square, got {} instead'.format(L.shape))
    return linear_problem(L, f, eps=eps)
