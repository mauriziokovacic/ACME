import numpy
import torch
from ..utility.isnumpy  import *
from ..utility.istorch  import *
from ..utility.isscalar import *
from .norm              import *
from .dot               import *
from .cross             import *


def tan(A, B=None, dim=1):
    """
    Computes the tangent value of the angle between the inputs along the specified dimension

    Parameters
    ----------
    A : Tensor or scalar
        the first input tensor
    B : Tensor (optional)
        the second input tensor (default is None)
    dim : int (optional)
        dimension along the tangent is computed, if B is a Tensor (default is 1)

    Returns
    -------
    Tensor or scalar
        the tensor (scalar) containing the tangent values
    """

    if B is None:
        if isnumpy(A):
            return numpy.tan(A)
        if istorch(A):
            return torch.tan(A)
        return math.tan(A)
    return norm(cross(A, B, dim=dim))/dot(A, B, dim=dim)
