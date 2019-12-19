import torch


class NoiseLayer(torch.nn.Module):
    """
    A class representing a layer returning a random noise tensor

    Attributes
    ----------
    dim : list or tuple
        the shape of the noise tensor
    device : str or torch.device
            the device to store the tensors to

    Methods
    -------
    forward(x)
        evaluates the module
    """

    def __init__(self, dim, device='cuda:0'):
        """
        Parameters
        ----------
        dim : list or tuple
            the shape of the noise tensor
        device : str or torch.device (optional)
            the device to store the tensors to (default is 'cuda:0')
        """

        super(NoiseLayer, self).__init__()
        self.dim = dim

    def forward(self, *args, **kwargs):
        """
        Returns a random noise tensor

        Returns
        -------
        Tensor
            the random noise tensor
        """

        return torch.rand(self.dim, dtype=torch.float, device=self.device)
