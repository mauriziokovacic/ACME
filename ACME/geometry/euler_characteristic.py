from ..utility.row import *
from ..utility.col import *

def euler_characteristic(P,E,F,V=None):
    """
    Returns the euler characteristic of the given set of points,edges,faces and volumes

    Parameters
    ----------
    P : Tensor
        the point set tensor
    E : LongTensor
        the edge tensor
    F : LongTensor
        the topology tensor
    V : LongTensor (optional)
        the volume tensor (default is None)

    Returns
    -------
    int
        the euler characteristic
    """

    return row(P)-col(E)+col(F)-col(V)
