import torch
from .model import *


class Cycle(Model):
    """
    A class representing the cycle model from:
        "Preventing self-intersection with cycle regularization in neural networks
         for mesh reconstruction from a single RGB image"

    Attributes
    ----------
    f : torch.nn.Module
        the decoder architecture
    g : torch.nn.Module
        the inverse decoder architecture

    Methods
    -------
    forward(x, s)
        returns the cycle model outputs
    """

    def __init__(self, f, g, name='Cycle', **kwargs):
        """
        Parameters
        ----------
        f : torch.nn.Module
            the decoder architecture
        g : torch.nn.Module
            the inverse decoder architecture
        name : str (optional)
            the name of the model
        """

        super(Cycle, self).__init__(name=name, **kwargs)
        self.f = f
        self.g = g
        self.add_module('f', self.f)
        self.add_module('g', self.g)

    def forward(self, x, s, detach=False):
        """
        Returns the cycle model outputs

        Parameters
        ----------
        x : object
            the input data
        s : object
            the signature data
        detach : bool (optional)
            if True detaches the input for the inverse network (default is False)

        Returns
        -------
        (object, object)
            the decoder and the inverse decoder outputs
        """
        y     = self.f(x, s)
        x_inv = self.g(y if not detach else y.detach(), s)
        return y, x_inv


def cycle_consistency_loss(x_hat, x_inv):
    """
    Returns the loss for the Cycle model

    Parameters
    ----------
    x_hat : Tensor
        the target tensor
    x_inv : Tensor
        the inverse tensor, output of the Cycle model

    Returns
    -------
    Tensor
        the (1,) loss tensor
    """

    return torch.sum(torch.pow(x_hat-x_inv, 2))
