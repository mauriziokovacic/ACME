from . import uniquetol

def unique(A,ByRows=False):
    """
    Returns the unique elements in A.

    This function returns a triplet in the form of (C,iA,iC), such that:
        - C = A[iA]
        - A = C[iC]

    Parameters
    ----------
    A : Tensor
        the input tensor
    ByRows : bool (optional)
        if True treats the rows as a single entity

    Returns
    -------
    (Tensor,LongTensor,LongTensor)
        returns the tensor of unique values/rows, their indices within the input tensor and the input tensor indices within the unique
    """

    return uniquetol(A,tol=0,ByRows=ByRows)
