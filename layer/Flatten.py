from torch.nn import Module


class Flatten(Module):
    """
    A layer performing the flattening of the input tensor

    Methods
    -------
    forward(input)
        flattens the input tensor
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        """
        Flattens the input tensor

        Parameters
        ----------
        input : Tensor
            the input tensor

        Returns
        -------
        Tensor
            the flattened input
        """

        return input.view(input.size(0), -1)
