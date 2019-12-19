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


class VariationalSampler(Model):
    """
    A class representing the sampler used in Variational AutoEncoders

    Attributes
    ----------
    mu : callable
        a function/torch.nn.Module used to compute the mean tensor
    sigma : callable
        a function/torch.nn.Module used to compute the standard deviation tensor

    Methods
    -------
    forward(y)
        returns the features extracted from the input tensor
    """

    def __init__(self,
                 mean_model=(lambda y: y[:, :y.size(1)//2]),
                 sdev_model=(lambda y: y[:, y.size(1)//2:]),
                 name='VariationalSampler', **kwargs):
        """
        Parameters
        ----------
        mean_model : callable (optional)
            a function/torch.nn.Module used to compute the mean tensor (default is f(y)=y[:,:n/2])
        sdev_model : callable (optional)
            a function/torch.nn.Module used to compute the standard deviation tensor (default is f(y)=y[:,n/2:])
        name : str (optional)
            the name for the Variational Sampler (default is 'VariationalSampler')
        """

        super(VariationalSampler, self).__init__(name=name, **kwargs)
        if isinstance(mean_model, torch.nn.Module):
            self.add_module('mu', mean_model)
        else:
            self.mu = mean_model
        if isinstance(sdev_model, torch.nn.Module):
            self.add_module('sigma', sdev_model)
        else:
            self.sigma = sdev_model

    def forward(self, y):
        """
        Returns the features extracted from the input tensor

        Parameters
        ----------
        y : Tensor
            the input tensor

        Returns
        -------
        (Tensor, Tensor, Tensor)
            the output features tensor, the mean tensor and the standard deviation tensor
        """

        mu    = self.mu(y)
        sigma = self.sigma(y)
        eps   = torch.empty_like(mu, dtype=mu.dtype, device=self.device).normal_()
        z     = mu + torch.exp(sigma/2)*eps
        return z, mu, sigma


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

    def __init__(self, encoder, decoder, z_sampler=VariationalSampler(), name='VariationalAutoEncoder', **kwargs):
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

        super(VariationalAutoEncoder, self).__init__(encoder, decoder, name=name, **kwargs)
        if isinstance(z_sampler, torch.nn.Module):
            self.add_module('z_sampler', z_sampler)
        else:
            self.z_sampler = z_sampler

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


class U_Net(AutoEncoder):
    """
    A class representing a U-Net

    Attributes
    ----------
    encoder : torch.nn.Module
        the encoder architecture
    decoder : torch.nn.Module
        the decoder architecture

    Methods
    -------
    forward(x)
        returns the U-Net output
    """

    def __init__(self, encoder, decoder, connection=None, name='U-Net', **kwargs):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            the encoder architecture
        decoder : torch.nn.Module
            the decoder architecture. Must contains HookLayers and the specified connection indices
        connection : LongTensor (optional)
            the (N,2,) indices tensor of the connected layers in encoders and decoders. If None if will be
            automatically computed (default is None)
        name : str (optional)
            the name of the U-Net (default is 'U-Net')
        """

        super(U_Net, self).__init__(encoder=encoder, decoder=decoder, name=name, **kwargs)
        if connection is None:
            connection = torch.cat((torch.arange(             0, len(encoder),  1).unsqueeze(1),
                                    torch.arange(len(decoder)-1,           -1, -1).unsqueeze(1),), dim=1)
        for i, j in connection:
            self.decoder[j].bind(self.encoder[i])

    def forward(self, x):
        """
        Returns the U-Net output

        Parameters
        ----------
        x : Tensor
            the U-Net input

        Returns
        -------
        Tensor
            the decoder output
        """

        return super(U_Net, self).forward(x)[0]


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


def KL_divergence(mu, sigma, beta=1):
    """
    Parameters
    ----------
    mu : Tensor
        the autoencoder mean tensor
    sigma : Tensor
        the autoencoder standard deviation tensor
    beta : float (optional)
        the disentanglement coefficient (default is 1)

    Returns
    -------
    Tensor
        the KL divergence tensor
    """

    return beta * torch.mean(0.5 * torch.sum(mu**2 + torch.exp(sigma) - sigma - 1, 1))


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

    return reconstruction_loss(x, x_hat) + KL_divergence(mu, sigma, beta=beta)
