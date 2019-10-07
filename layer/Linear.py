from .Layer import *


class Linear(Layer):
    """
    A class representing a fully connected linear layer, followed by activation and batch normalization
    """

    def __init__(self, *args, activation=None, batch_norm=None, **kwargs):
        """
        Parameters
        ----------
        layer : torch.nn.Module
            the layer to evaluate
        activation : torch.nn.Module (optional)
            the layer activation function (default is None)
        batch_norm : torch.nn.Module (optional)
            the layer batch normalization (default is None)
        pooling : torch.nn.Module (optional)
            the layer pooling (default is None)
        """

        super(Linear, self).__init__(
            torch.nn.Linear(*args, **kwargs),
            activation=activation,
            batch_norm=batch_norm,
            pooling=None
        )
