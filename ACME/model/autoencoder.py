import torch
from .model import *


class AutoEncoder(Model):
    """
    A class representing a standard autoencoder

    Attributes
    ----------
    encoder : torch.nn.Module
        the encoder architecture
    decoder : torch.nn.Module
        the decoder architecture

    Methods
    -------
    forward(x)
        returns the autoencoder output
    """

    def __init__(self, encoder, decoder, name='AutoEncoder', **kwargs):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            the encoder architecture
        decoder : torch.nn.Module
            the decoder architecture
        name : str (optional)
            the name of the autoencoder (default is 'AutoEncoder')
        """

        super(AutoEncoder, self).__init__(name=name, **kwargs)
        self.add_module('encoder', encoder)
        self.add_module('decoder', decoder)

    def forward(self, x):
        """
        Returns the autoencoder output

        Parameters
        ----------
        x : Tensor
            the autoencoder input

        Returns
        -------
        (Tensor, Tensor)
            the decoder and encoder output
        """

        y     = self.encoder(x)
        x_hat = self.decoder(y)
        return x_hat, y




def reconstruction_loss(x, x_hat, *args, **kwargs):
    """
    Returns the binary cross entropy loss

    Parameters
    ----------
    x : Tensor
        the autoencoder input tensor
    x_hat : Tensor
        the autoencoder output tensor

    Returns
    -------
    Tensor
        the (1,) loss tensor
    """

    return torch.nn.BCELoss()(x_hat, x)


