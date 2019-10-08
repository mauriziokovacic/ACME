from .Layer import *


class Conv(Layer):
    """
    A class representing a generic convolutional layer over 1D, 2D or 3D tensors
    """

    def __init__(self, dim, *args, activation=None, batch_norm=None, pooling=None, **kwargs):
        """
        Parameters
        ----------
        dim : int
            the dimension of the convolution. Accepted dims are 1, 2 or 3
        *args : ...
            the convolution layer arguments
        activation : torch.nn.Module (optional)
            the layer activation function (default is None)
        batch_norm : torch.nn.Module (optional)
            the layer batch normalization (default is None)
        pooling : torch.nn.Module (optional)
            the layer pooling (default is None)
        **kwargs : ...
            the convolution layer keyword arguments
        """
        super(Conv, self).__init__(
            eval('torch.nn.Conv'+str(dim)+'d(*args, **kwargs)'),
            activation=activation,
            batch_norm=batch_norm,
            pooling=pooling,
            dropout=None,
        )
