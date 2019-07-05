from .to_skinny import *
from .flatten   import *


def to_column(*tensors):
    """
    Transforms the input tensor into a column vector

    If the input tensor is a matrix, it will be flattened into a (N,1,) vector
    If the input tensor is a (N,) vector, it will be unsqueezed in dimension 1
    No operation is performed if the input is already a (N,1,) vector

    Parameters
    ----------
    *tensors : Tensor...
        a sequence of (N,) or (M,N,) tensors

    Returns
    -------
    Tensor or list
        a (N,1,) or (N,M,) tensor, input if input is (N,1,)

    Raises
    ------
    AssertionError
        if inputs are not tensors
    """

    out = [to_skinny(flatten(t)) for t in tensors]
    return out if len(out) > 1 else out[0]
