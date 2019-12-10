import torch


def dq_imag(DQ, dim=-1):
    """
    Returns the imaginary part of the given dual quaternions

    Parameters
    ----------
    DQ : Tensor
        a (8,) or (N,8,) dual quaternion tensor
    dim : int (optional)
        the dimension along the extraction is performed (default is -1)

    Returns
    -------
    Tensor
        the (4,) or (N,4,) quaternion tensor
    """

    return torch.index_select(DQ, dim, torch.tensor(range(4, 8))).squeeze()
