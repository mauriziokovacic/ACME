import torch

def repmat(tensor,*size):
    """
    Repeats the tensor along its dimensions by the given times

    Example:
        repmat([[1,2,3]],1,2) -> [[1,2,3,1,2,3]]

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    *size : int...
        a sequence of times to repeats the tensor along a particular dimension

    Returns
    -------
    Tensor
        a tensor equivalent to concatenating the input along some dimension
    """

    return tensor.repeat(*size)
