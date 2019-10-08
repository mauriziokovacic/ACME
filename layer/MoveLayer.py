import torch


class MoveLayer(torch.nn.Module):
    """
    A layer moving the input to a specific device

    Attributes
    ----------
    device : str or torch.device
        the device to move the input to

    Methods
    -------
    forward(input)
        returns the moved input
    """

    def __init__(self, device):
        super(MoveLayer, self).__init__()
        """
        Parameters
        ----------
        device : str or torch.device
            the device to move the input to
        """

        super(MoveLayer, self).__init__()
        self.device = device

    def forward(self, input):
        """
        Returns the moved input

        Parameters
        ----------
        input : object
            the input object

        Returns
        -------
        object
            the moved input
        """

        return input.to(device=self.device)
