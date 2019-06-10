from ACME.utility.circshift import *

def polyshift(T,k):
    """
    Shifts the nodes within a polygon k times

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    k : int
        the times to shift

    Returns
    -------
    LongTensor
        the shifted topology tensor
    """

    return circshift(T,k,dim=0)
