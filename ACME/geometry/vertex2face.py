from .barycenter import *


def vertex2face(T, vertex_data):
    """
    Returns the face data computed from the vertices

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    vertex_data : Tensor
        the vertex data

    Returns
    -------
    Tensor
        the face data
    """

    return barycenter(vertex_data, T)
