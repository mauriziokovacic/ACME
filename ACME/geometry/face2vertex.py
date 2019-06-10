from ACME.utility.accumarray import *
from ACME.topology.poly2lin  import *

def face2vertex(T,face_data):
    """
    Returns the vertex data computed from the faces

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    face_data : Tensor
        the face data

    Returns
    -------
    Tensor
        the vertex data
    """

    I,t = poly2lin(T)
    return accumarray(I,face_data[t]) / accumarray(I,1)
