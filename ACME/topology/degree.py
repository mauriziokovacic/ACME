import torch

def degree(A):
    """
    Returns the degree matrix from the given adjacency matrix

    Parameters
    ----------
    A : Tensor
        the input adjacency matrix

    Returns
    -------
    Tensor
        the degree matrix

    Raises
    ------
    AssertionError
        if input is not a matrix
    """

    assert ismatrix(A), 'Tensor must be a matrix'
    return torch.diag(torch.sum(A,1,keepdim=False))
