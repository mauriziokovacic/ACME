from math.cross   import *
from math.normvec import *

def triangle_normal(P,T,dim=1):
    """
    Returns the normal of the input triangles.

    Parameters
    ----------
    P : Tensor
        the input point set tensor
    T : LongTensor
        the topology tensor
    dim : int (optional)
        the dimension along the normals are computed (default is 1)

    Returns
    -------
    Tensor
        the normals of the input triangulation
    """

    Pi,Pj,Pk = P[T]
    return normvec(cross(Pj-Pi,Pk-Pi,dim),dim=dim)
