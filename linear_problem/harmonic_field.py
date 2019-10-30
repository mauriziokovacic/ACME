from .add_constraints  import *
from .poisson_equation import *


def harmonic_field(L, k, i):
    """
    Computes the harmonic field from the given data, using the i hard constraints

    Parameters
    ----------
    L : Tensor or SparseTensor
        the (N,N,) Laplacian matrix
    k : Tensor
        the (N,D,) known values tensor
    i : LongTensor
        the indices of the hard constraints

    Returns
    -------
    Tensor
        the (N,D,) solution tensor
    """

    return poisson_equation(add_constraints(L, i), k, eps=0)
