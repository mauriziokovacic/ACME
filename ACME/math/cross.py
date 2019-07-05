import torch


def cross(A, B, dim=1):
    """
    Computes the cross product of the input tensors along the specified dimension

    Parameters
    ----------
    A : Tensor
        first input tensor
    B : Tensor
        second input tensor
    dim : int (optional)
        dimension along the cross product is performed

    Returns
    -------
    Tensor
        the cross products of the input tensors
    """

    return torch.cross(A, B, dim=dim)
