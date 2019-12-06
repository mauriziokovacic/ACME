from ..utility.uniquetol import *


def soup2mesh(P, T):
    """
    Converts a polygon soup into a mesh

    Parameters
    ----------
    P : Tensor
        the input points set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    (Tensor,LongTensor,LongTensor,LongTensor)
        the unique points, the new topology tensor, the indices of the unique points and the mapping of the old points
    """

    I, J = uniquetol(P, tol=0.0001, ByRows=True)[1:]
    return P[I], J[T], I, J
