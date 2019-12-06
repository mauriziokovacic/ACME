from ..math.normvec import *
from ..math.angle   import *


def triangle_angle(P, T):
    """
    Returns the three angles of the given triangles

    Parameters
    ----------
    P : Tensor
        the input point set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    (Tensor,Tensor,Tensor)
        the angles of the input triangles
    """

    Pi, Pj, Pk = P[T]
    Eij = normr(Pj-Pi)
    Ejk = normr(Pk-Pj)
    Eki = normr(Pi-Pk)
    return angle(Eij, -Eki), angle(Ejk, -Eij), angle(Eki, -Ejk)
