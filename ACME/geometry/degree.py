import torch

def Degree(M):
    """
    Returns the degree matrix from the given matrix

    Parameters
    ----------
    M : Tensor
        the input matrix

    Returns
    -------
    Tensor
        the degree matrix

    Raises
    ------
    AssertionError
        if input is not a matrix
    """

    assert ismatrix(M), 'Tensor must be a matrix'
    D = torch.sum(M,1,keepdim=False)
    return torch.diag(D)
