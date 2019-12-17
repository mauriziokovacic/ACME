import torch


def channel(C, index, dim=-1):
    """
    Extracts the specified channel from the given colors along a given dimension

    Parameters
    ----------
    C : Tensor
        the (C,W,H,) or (N,C,) color tensor
    index : int
        the index of the channel to extract
    dim : int (optional)
        the dimension along the operation is performed (default is -1)

    Returns
    -------
    Tensor
        the (1,W,H,) or (N,1,) color channel tensor
    """

    return torch.index_select(C, dim, torch.tensor([*index], C.device))
