import torch
from ..layer.VariationalSampler import *
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

    def __init__(self, encoder, decoder, name='AutoEncoder'):
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

        super(AutoEncoder, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder
        self.add_module('encoder', self.encoder)
        self.add_module('decoder', self.decoder)

    def forward(self, x):
        """
        Returns the autoencoder output

        Parameters
        ----------
        x : Tensor
            the autoencoder input

        Returns
        -------
        Tensor
            the autoencoder output
        """

        y     = self.encoder(x)
        x_hat = self.decoder(y)
        return x_hat



class VariationalAutoEncoder(AutoEncoder):
    """
    A class representing a variational autoencoder

    Attributes
    ----------
    z_sampler : callable



    Methods
    -------
    forward(x)
        returns the autoencoder output
    """

    def __init__(self, encoder, decoder, z_sampler=VariationalSampler(), name='VariationalAutoEncoder'):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            the encoder architecture
        decoder : torch.nn.Module
            the decoder architecture
        z_sampler : callable (optional)
            the z tensor sampler (default is VariationalSampler())
        name : str (optional)
            the name of the autoencoder (default is 'VariationalAutoEncoder')
        """

        super(VariationalAutoEncoder, self).__init__(encoder, decoder, name=name)
        self.z_sampler = sampler



    def forward(self, x):
        """
        Returns the autoencoder output

        Parameters
        ----------
        x : Tensor
            the autoencoder input

        Returns
        -------
        (Tensor, Tensor, Tensor)
            the autoencoder output, its mean tensor and its standard deviation tensor
        """

        y            = self.encoder(x)
        z, mu, sigma = self.z_sampler(y)
        x_hat        = self.decoder(z)
        return x_hat, mu, sigma



def AELoss(x, x_hat, *args, **kwargs):
    """
    Returns the standard loss for an autoencoder

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

    return torch.mean(torch.pow(x-x_hat,2))



def VAELoss(x, x_hat, mu, sigma, beta=1):
    """
    Returns the loss for a variational autoencoder

    Parameters
    ----------
    x : Tensor
        the autoencoder input tensor
    x_hat : Tensor
        the autoencoder output tensor
    mu : Tensor
        the autoencoder mean tensor
    sigma : Tensor
        the autoencoder standard deviation tensor
    beta : float (optional)
        the disentanglement coefficient (default is 1)

    Returns
    -------
    Tensor
        the (1,) loss tensor
    """

    margin = torch.sum(x * torch.log(x_hat) + (1-x) * torch.log(1-x_hat), 1)
    KL_div = 0.5 * torch.sum(mu**2 + sigma**2 - torch.log(1e-8 + sigma**2) - 1, 1)
    loss   = torch.mean(margin) - beta*torch.mean(KL_div)
    return -loss
