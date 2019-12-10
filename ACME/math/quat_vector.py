import torch


def quat_vector(Q, dim=-1):
    """
    Returns the vector part of the given quaternions

    Parameters
    ----------
    Q : Tensor
        the (4,) or (N,4,) quaternion tensor
    dim : int (optional)
        the dimension to extract the scalar part from (default is -1)

    Returns
    -------
    Tensor
        the (3,) or (N,3,) quaternion scalar part
    """

    return torch.index_select(Q, dim, torch.tensor(range(Q.shape[dim]-1))).squeeze()
