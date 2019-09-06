from .istensor  import *
from .ndim      import *
from .transpose import *
from .unsqueeze import *
from .isfat     import *


def to_skinny(tensor):
    """
    Transforms the input tensor into a skinny matrix

    If the input tensor is a fat matrix, its transposed will be returned
    If the input tensor is a vector (n,), it will be unsqueezed in dimension 1
    No operation is performed if the input is already skinny

    Parameters
    ----------
    tensor : Tensor
        a n or nxm tensor

    Returns
    -------
    Tensor
        a nx1 or mxn tensor, input if input is skinny

    Raises
    ------
    AssertError
        if input is not a tensor
    """

    assert istensor(tensor), 'Input must be a valid tensor'
    if ndim(tensor) == 1:
        return unsqueeze(tensor, 1)
    if isfat(tensor):
        return transpose(tensor)
    return tensor
