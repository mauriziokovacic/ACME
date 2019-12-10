import torch


def quat_eye(n=1, dtype=torch.float, device='cuda:0'):
    """
    Returns n identity quaternions

    Parameters
    ----------
    n : int (optional)
        the number of quaternion to return (default is 1)
    dtype : type (optional)
        the data type of the output tensor (default is torch.float)
    device : str or torch.device (optional)
        the device to store the tensors to (default is 'cuda:0')

    Returns
    -------
    Tensor
        the (4,) or (N,4,) quaternion tensor
    """
    return torch.tensor([1, 0, 0, 0], dtype=dtype, device=device).repeat(n, 1).squeeze()
