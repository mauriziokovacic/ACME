from ..utility.matpow import *
from ..math.speye     import *


def taubin_smoothing(K, P, alpha=0.4, mu=-0.6, iter=1):
    """
    Smooths the given mesh using the implicit Taubin smoothing

    Parameters
    ----------
    K : Tensor or SparseTensor
        the (N,N,) weight matrix
    P : Tensor
        the (N,D,) data tensor
    alpha : float (optional)
        the first smoothing factor (default is 0.4)
    mu : float (optional)
        the second smoothing factor (default is -0.6)
    iter : int (optional)
        number of smoothing iterations (default is 1)

    Returns
    -------
    Tensor
        the (N,D,) smoothed data tensor

    Raises
    ------
    AssertionError
        if condition 0 < alpha < -mu is not met
    """

    assert (alpha > 0) and (-mu > alpha), 'Smoothing parameters must satisfy 0 < alpha < -mu'
    I = speye_like(K)
    return matmul(matpow(matmul(I-mu*K, I-alpha*K), iter), P)
