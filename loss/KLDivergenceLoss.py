from .Loss import *


class KLDivergenceLoss(Loss):
    """
    A class representing the KL divergence for variational autoencoder

    Attributes
    ----------
    beta : float
        the KL divergence disentanglement factor
    """

    def __init__(self, *args, beta=1, name='KL_div', **kwargs):
        """
        Parameters
        ----------
        beta : float (optional)
            the KL divergence disentanglement factor (default is 1)
        """

        super(KLDivergenceLoss, self).__init__(*args, name=name, **kwargs)
        self.beta = beta

    def __eval__(self, input, output):
        """
        Returns the KL divergence for the given input

        Parameters
        ----------
        input : ...
            unused parameter
        output : list or tuple
            a pair containing mu and sigma of the variational autoencoder

        Returns
        -------
        Tensor
            the (1,) loss tensor
        """
        mu    = output[0]
        sigma = output[1]
        return self.beta * torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(sigma) - sigma - 1, 1))
