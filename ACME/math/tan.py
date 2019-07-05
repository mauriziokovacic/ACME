from .norm  import *
from .dot   import *
from .cross import *


def tan(A, B, dim=1):
    """
    Computes the tangent value of the angle between the inputs along the specified dimension

    Parameters
    ----------
    A : Tensor
        the first input tensor
    B : Tensor
        the second input tensor
    dim : int (optional)
        dimension along the tangent is computed

    Returns
    -------
    Tensor
        the tensor containing the tangent values
    """
    return norm(cross(A, B, dim=dim))/dot(A, B, dim=dim)
