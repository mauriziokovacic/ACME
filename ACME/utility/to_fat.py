from .istensor  import *
from .ndim      import *
from .transpose import *
from .unsqueeze import *
from .isskinny  import *

def to_fat(tensor):
    """
    Transforms the input tensor into a fat matrix

    If the input tensor is a skinny matrix, its transposed will be returned
    If the input tensor is a vector (n,), it will be unsqueezed in dimension 0
    No operation is performed if the input is already fat

    Parameters
    ----------
    tensor : Tensor
        a n or nxm tensor

    Returns
    -------
    Tensor
        a 1xn or mxn tensor, input if input is fat

    Raises
    ------
    AssertError
        if input is not a tensor
    """

    assert istensor(tensor), 'Input must be a valid tensor'
    if ndim(tensor) == 1:
        return unsqueeze(tensor, 0)
    if isskinny(tensor):
        return transpose(tensor)
    return tensor
