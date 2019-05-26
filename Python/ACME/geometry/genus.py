from .euler_characteristic import *

def genus(P,E,F,V=None):
    """
    Returns the genus of the input shape

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
        the genus of the input shape
    """

    return euler_characteristic(P,E,F,V)-2
