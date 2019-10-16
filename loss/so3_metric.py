import torch


def so3_metric(M):
    """
    Computes the SO(3) metric for the given matrix

    Parameters
    ----------
    M : Tensor
        a (3,3,) or (N,3,3,) tensor

    Returns
    -------
    Tensor
        the (1,) metric tensor
    """

    return torch.mean(torch.norm(torch.matmul(M, torch.transpose(M, -1, -2))-torch.eye(3), dim=(-2, -1), keepdim=True))
