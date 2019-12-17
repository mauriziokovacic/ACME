from .matmul import *
from .issquare import *


def matpow(M, exp):
    """
    Computes the matricial M^exp operation

    Parameters
    ----------
    M : Tensor
        the (N,N,) input matrix
    exp : int
        the power exponent

    Returns
    -------
    Tensor
        the (N,N,) output matrix

    Raises
    ------
    RuntimeError
        if M is not a square matrix
    """

    if not issquare(M):
        raise RuntimeError('Input matrix should be squared.')
    out = M
    for i in range(exp):
        out = matmul(M, out)
    return out
