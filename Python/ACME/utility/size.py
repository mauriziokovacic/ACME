def size(A):
    """
    Returns the size of each dimension of the input Tensor

    Parameters
    ----------
    A : Tensor
        A tensor/matrix

    Returns
    -------
    list
        the size of each dimension of the input tensor
    """

    if A is None:
        return 0
    return A.shape
