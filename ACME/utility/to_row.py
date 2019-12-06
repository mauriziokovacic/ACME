from .to_fat  import *
from .flatten import *


def to_row(*tensors):
    """
    Transforms the input tensor into a row vector

    If the input tensor is a matrix, it will be flattened into a (1,N,) vector
    If the input tensor is a (N,) vector, it will be unsqueezed in dimension 0
    No operation is performed if the input is already a (1,N,) vector

    Parameters
    ----------
    *tensors : Tensor...
        a sequence of (N,) or (N,M) tensors

    Returns
    -------
    Tensor or list
        a (1,N,) or (M,N,) tensor, input if input is (1,N,)

    Raises
    ------
    AssertionError
        if inputs are not tensors
    """

    out = [to_fat(flatten(t)) for t in tensors]
    return out if len(out) > 1 else out[0]
