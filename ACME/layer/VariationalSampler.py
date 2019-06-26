import torch

class VariationalSampler(torch.nn.Module):
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
                 mean_model=(lambda y: y[:,:y.size(1)]),
                 sdev_model=(lambda y: y[:,y.size(1):])):
        """
        Parameters
        ----------
        mean_model : callable (optional)
            a function/torch.nn.Module used to compute the mean tensor (default is f(y)=y[:,:n/2])
        sdev_model : callable (optional)
            a function/torch.nn.Module used to compute the standard deviation tensor (default is f(y)=y[:,n/2:])
        """

        super(VariationalSampler, self).__init__()
        self.mu    = mean_model
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
        eps   = torch.empty_like(mu, dtype=mu.dtype, device=mu.device).normal_()
        z     = mu + torch.exp(sigma/2)*eps
        return z, mu, sigma
