from ACME.utility.row import *
from ACME.utility.col import *

def affine2rotm(A):
    """
    Returns the rotation matrix M within the affine matrix.

    The affine matrix is represented as:
       A = |M 0|
           |0 1|
    where 0 is a zero row/column vector and 1 is a single value

    Parameters
    ----------
    M : Tensor
        a (4,4) or (3,3) matrix

    Returns
    -------
    Tensor
        a (3,3) or (2,2) matrix
    """

    return A[0:row(A),0:col(A)]
