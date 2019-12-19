from .autoencoder import *


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
