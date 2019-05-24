import torch

def dot(A,B,dim=1):
    """
    Computes the dot product of the input tensors along the specified dimension

    Parameters
    ----------
    A : Tensor
        first input tensor
    B : Tensor
        second input tensor
    dim : int (optional)
        dimension along the dot product is computed (default is 1)

    Returns
    -------
    Tensor
        the tensor containing the dot products
    """

    return torch.sum(A*B,dim,keepdim=True)
