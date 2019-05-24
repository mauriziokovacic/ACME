def poly2ind(T):
    """
    Extracts indices from the topology tensor

    Parameters
    ----------
    T : LongTensor
        the polygon tensor

    Returns
    -------
    (LongTensor,...)
        the indices tensors
    """

    return tuple(T)



def edge2ind(E):
    """
    Extracts indices from the edge tensor

    Parameters
    ----------
    E : LongTensor
        the edge tensor

    Returns
    -------
    (LongTensor,LongTensor)
        the tuple with indices tensors
    """

    return poly2ind(E)



def tri2ind(T):
    """
    Extracts indices from the triangle tensor

    Parameters
    ----------
    T : LongTensor
        the triangle tensor

    Returns
    -------
    (LongTensor,LongTensor, LongTensor)
        the tuple with indices tensors
    """

    return poly2ind(T)



def quad2ind(T):
    """
    Extracts indices from the quad tensor

    Parameters
    ----------
    T : LongTensor
        the quad tensor

    Returns
    -------
    (LongTensor,LongTensor,LongTensor,LongTensor)
        the tuple with indices tensors
    """

    return poly2ind(T)
