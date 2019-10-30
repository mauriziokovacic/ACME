from .poisson_equation import *


def laplacian_smoothing(L, P, alpha=0.5):
    """
    Smooths the given mesh using the implicit laplacian smoothing

    Parameters
    ----------
    L : Tensor or SparseTensor
        the (N,N,) Laplacian matrix
    P : Tensor
        the (N,D,) data tensor
    alpha : float (optional)
        the smoothing strenght (default is 0.5)

    Returns
    -------
    Tensor
        the (N,D,) smoothed data tensor
    """

    return poisson_equation(speye_like(L) + alpha * L, P, eps=0)
