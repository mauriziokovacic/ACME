from .poisson_equation import *


def heat_diffusion(A, t, L, k, eps=0.0001):
    """
    Computes the heat diffusion equation

    Parameters
    ----------
    A : Tensor or SparseTensor
        the (N,N,) density matrix
    t : float
        the diffusion time
    L : Tensor or SparseTensor
        the (N,N,) Laplacian matrix
    k : Tensor
        the (N,D,) initial heat tensor
    eps : float (optional)
        a regularizer value (default is 0.0001)

    Returns
    -------
    Tensor
        the (N,D,) heat tensor
    """

    return poisson_equation(A+t*L, k, eps=eps)
