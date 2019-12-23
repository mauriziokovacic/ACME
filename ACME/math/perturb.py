import torch


def perturb(tensor):
    """
    Perturbs the input tensor with random noise

    Parameters
    ----------
    tensor : Tensor
        the input tensor

    Returns
    -------
    Tensor
        the perturbed tensor
    """

    return torch.randn_like(tensor) * tensor + torch.randn_like(tensor)
