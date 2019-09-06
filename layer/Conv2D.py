import torch


class Conv2D(torch.nn.Module):
    """
    A wrapper class decomposing a square 2D convolution into 2 1D convolutions,
    effectively saving three parameters per kernel

    Attributes
    ----------
    model : torch.nn.Module
        the hidden architecture of the convolution

    Methods
    -------
    forward(input)
        Applies the 2D convolution to the input tensor
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        """
        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : int
            the number of kernels
        kernel_size : int
            the size of the kernel side
        **kwargs : ...
            the optional arguments for a torch.nn.Conv2d
        """

        super(Conv2D, self).__init__()
        self.model = torch.nn.Sequential(
                        torch.nn.Conv2d( in_channels, out_channels, kernel_size=(kernel_size, 1), **kwargs),
                        torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size), **kwargs),
                     )

    def forward(self, input):
        """
        Applies the 2D convolution to the input tensor

        Parameters
        ----------
        input : Tensor
            the input tensor

        Returns
        -------
        Tensor
            the convolved tensor
        """

        return self.model(input)
