from .norm import *
from .dot  import *

def cot(A,B,dim=1):
    """
    Computes the cotangent of the angle between the inputs along the specified dimension

    Parameters
    ----------
    A : Tensor
        the first input tensor
    B : Tensor
        the second input tensor
    dim : int (optional)
        dimension along the cotanget is computed

    Returns
    -------
    Tensor
        the tensor containing the cotangent values
    """

    return dot(A,B,dim=dim)/norm(cross(A,B,dim=dim))
