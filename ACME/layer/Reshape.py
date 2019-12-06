import torch


class Reshape(torch.nn.Module):
    """
    A layer performing the reshape of the input tensor

    Attributes
    ----------
    dim : int...
        the new shape of the tensor

    Methods
    -------
    forward(input)
        reshapes the input tensor
    """

    def __init__(self, dim):
        """
        Parameters
        ----------
        dim : int...
            the new shape of the tensor
        """

        super(Reshape, self).__init__()
        self.dim = (-1,)+dim

    def forward(self, input):
        """
        Reshapes the input tensor

        Parameters
        ----------
        input : Tensor
            the input tensor

        Returns
        -------
        Tensor
            the reshaped input
        """

        return torch.reshape(input, self.dim)
