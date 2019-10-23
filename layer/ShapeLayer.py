import torch


class ShapeLayer(torch.nn.Module):
    """
    A class defining a constant layer that returns a given shape

    Methods
    -------
    forward(*args, **kwargs)
        returns the given shape
    """

    def __init__(self, shape):
        """
        Parameters
        ----------
        shape : Data or Tensor
            the shape to return
        """

        self.shape = shape

    def forward(self, *args, **kwargs):
        """
        Returns the given shape

        Parameters
        ----------
        args : ...
        kwargs : ...

        Returns
        -------
        Data or Tensor
            the given shape
        """
        return self.shape.clone()
