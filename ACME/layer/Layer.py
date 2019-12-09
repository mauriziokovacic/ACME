import torch


class Layer(torch.nn.Module):
    """
    A class representing a generic layer followed by activation, batch normalization and pooling

    Attributes
    ----------
    layer : torch.nn.Module
        the layer model

    Methods
    -------
    forward(*args, **kwargs)
        returns the layer output
    """

    def __init__(self, layer, activation=None, batch_norm=None, pooling=None, dropout=None):
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

        super(Layer, self).__init__()
        l = [layer]
        if activation is not None:
            l += [activation]
        if batch_norm is not None:
            l += [batch_norm]
        if pooling is not None:
            l += [pooling]
        if dropout is not None:
            l += [dropout]
        layer = torch.nn.Sequential(*l)
        self.add_module('layer', layer)

    def forward(self, *args, **kwargs):
        """
        Returns the layer output

        Parameters
        ----------
        args : ...
        kwargs : ...

        Returns
        -------
        ...
            the layer output
        """

        return self.layer(*args, **kwargs)
