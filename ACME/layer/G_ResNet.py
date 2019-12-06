import torch
from ..utility.isdict     import *
from ..utility.isfunction import *
from ..utility.islist     import *
from ..utility.isstring   import *
from ..utility.strrep     import *
from .HookLayer           import GResidualLayer


class G_ResNet(torch.nn.Module):
    """
    A class representing a G-ResNet.
    The network has a similar behaviour of a torch.nn.Sequential, but contains residual layers

    Attributes
    ----------
    net : torch.nn.ModuleList
        the GCNConv layers composing the ResNet
    activation : torch.nn.Module
        the activation function to be applied to each layer

    Methods
    -------
    forward(x, edge_index, **kwargs)
        returns the G-ResNet output
    """

    def __init__(self,
                 cls,
                 in_channels,
                 out_channels,
                 adjacency={},
                 operation='cat',
                 dim=1,
                 activation=torch.nn.ReLU(inplace=False),
                 **kwargs):
        """
        Parameters
        ----------
        cls : class
            the class of the contained layers
        in_channels : int
            the number of input channels
        out_channels : list
            the output channels, one int for each layer
        adjacency : dict (optional)
            the adjacency for residual layers.
            The key is the index of the layer and the value is a list of indices of the connected layers,
            but the previous (default is {})
        operation : str or callable or list or dict (optional)
            the residual operation to be performed on the i-th graph node (default is 'cat')
        dim : int or list or dict (optional)
            the dimension along the residual operation is performed on the i-th graph node (default is 1)
        activation : torch.nn.Module (optional)
            an activation module to be applied to all the layers (default is torch.nn.ReLU)
        kwargs : ...
            the keyword arguments to be fed to the layers constructor
        """

        super(G_ResNet, self).__init__()
        if not islist(out_channels):
            out_channels = [out_channels]
        self.add_module('net', torch.nn.ModuleList())
        self.net.append(cls(in_channels, out_channels[0], **kwargs))
        ic = out_channels[0]
        for n in range(1, len(out_channels)):
            # Check if is a residual layer
            if n in adjacency:
                p = adjacency[n]
                # Check the operation
                if isstring(operation) or isfunction(operation):
                    op = operation
                elif islist(operation) or isdict(operation):
                    op = operation[n]
                else:
                    raise ValueError('Operation value is invalid')
                # Check the dimension
                if islist(dim) or isdict(dim):
                    d = dim[n]
                else:
                    d = dim
                # Compute the input channels
                if op == 'cat':
                    for j in p:
                        ic += out_channels[j]
                else:
                    ic = out_channels[p[0]]
                # Add the new layer
                self.net.append(GResidualLayer(cls(ic, out_channels[n], **kwargs),
                                               operation=op,
                                               dim=d,
                                               hook_layer=[self.net[j] for j in p]))
            else:
                self.net.append(cls(ic, out_channels[n], **kwargs))
            ic = out_channels[n]
        self.activation = activation

    def forward(self, x, *args, **kwargs):
        """
        Returns the G-ResNet output.
        The layers are evaluated sequentially

        Parameters
        ----------
        x : Tensor
            the input tensor
        args : ...
            the arguments
        kwargs : ...
            the keyword arguments

        Returns
        -------
        Tensor
            the output of the net
        """

        y = x
        for layer in self.net:
            y = self.activation(layer(y, *args, **kwargs))
        return y

    def is_empty(self):
        """
        Returns True if the network has no layer

        Returns
        -------
        bool
            True if the network has no layer, False otherwise
        """

        return self.size() == 0

    def size(self):
        """
        Returns the number of layers in the network

        Returns
        -------
        int
            the number of layers in the network
        """

        return len(self.net)

    def __repr__(self):
        return strrep(self.net.__repr__(), 'ModuleList', 'G-ResNet')

    def __len__(self):
        return self.size()

    def __getitem__(self, i):
        if isinstance(self.net[i], GResidualLayer):
            return self.net[i].layer
        return self.net[i]
