from ..math.cot import *

def triangle_cotangent(P,T):
    """
    Returns the angles cotangents of the given triangles

    Parameters
    ----------
    P : Tensor
        the input point set tensor
    T : LongTensor
        the topology tensor

    Returns
    -------
    (Tensor,Tensor,Tensor)
        the cotangent tensors of the input triangles
    """

    Pi,Pj,Pk = P[T]
    Eij = Pj-Pi
    Ejk = Pk-Pj
    Eki = Pi-Pk
    return cot(Eij,-Eki),cot(Ejk,-Eij),cot(Eki,-Ejk)
