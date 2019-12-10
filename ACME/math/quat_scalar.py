import torch


def quat_scalar(Q, dim=-1):
    """
    Returns the scalar part of the given quaternions

    Parameters
    ----------
    Q : Tensor
        the (4,) or (N,4,) quaternion tensor
    dim : int (optional)
        the dimension to extract the scalar part from (default is -1)

    Returns
    -------
    Tensor
        the (N,) quaternion scalar part
    """

    return torch.index_select(Q, dim, torch.tensor(Q.shape[dim]-1)).squeeze()
