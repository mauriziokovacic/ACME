from ..utility.matmul import *
from .poisson_equation import *


def taubin_smoothing(K, P, alpha=0.5, mu=0.5):
    """
    Smooths the given mesh using the implicit Taubin smoothing

    Parameters
    ----------
    L : Tensor or SparseTensor
        the (N,N,) Laplacian matrix
    P : Tensor
        the (N,D,) data tensor
    alpha : float (optional)
        the first smoothing factor (default is 0.5)
    mu : float (optional)
        the second smoothing factor (default is 0.5)

    Returns
    -------
    Tensor
        the (N,D,) smoothed data tensor
    """

    I = speye_like(K)
    return poisson_equation(matmul(I-alpha*K, I-mu*K), P, eps=0)
