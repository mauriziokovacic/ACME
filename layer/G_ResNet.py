import torch
from torch_geometric.nn   import GCNConv
from ..utility.isdict     import *
from ..utility.isfunction import *
from ..utility.islist     import *
from ..utility.isstring   import *
from ..utility.strrep     import *
from .HookLayer           import ResidualLayer


class G_ResNet(torch.nn.Module):
    """
    A class representing a G-ResNet.
    The network has a similar behaviour of a torch.nn.Sequential, but contains residual layers

    Attributes
    ----------
    net : torch.nn.ModuleList
        the GCNConv layers composing the ResNet

    Methods
    -------
    forward(x, edge_index, **kwargs)
        returns the G-ResNet output
    """

    def __init__(self, in_channels, out_channels, adjacency, operation='cat', dim=1):
        """
        Parameters
        ----------
        in_channels : int
            the number of input channels
        out_channels : list
            the output channels, one int for each layer
        adjacency : dict
            the adjacency for residual layers.
            The key is the index of the layer and the value is a list of indices of the connected layers
        operation : str or callable or list or dict (optional)
            the residual operation to be performed on the i-th graph node (default is 'cat')
        dim : int or list or dict (optional)
            the dimension along the residual operation is performed on the i-th graph node (default is 1)
        """

        super(G_ResNet, self).__init__()
        self.add_module('net', torch.nn.ModuleList())
        self.net.append(GCNConv(in_channels, out_channels[0]))
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
                    ic = 0
                    for j in p:
                        ic += out_channels[j]
                else:
                    ic = out_channels[p[0]]
                # Add the new layer
                self.net.append(ResidualLayer(GCNConv(ic, out_channels[n]),
                                              operation=op,
                                              dim=d,
                                              hook_layer=[self.net[j] for j in p]))
            else:
                self.net.append(GCNConv(ic, out_channels[n]))
            ic = out_channels[n]

    def forward(self, x, edge_index, **kwargs):
        """
        Returns the G-ResNet output.
        The layers are evaluated sequentially

        Parameters
        ----------
        x : Tensor
            the input tensor
        edge_index : LongTensor
            the (2,N,) edge tensor
        kwargs : ...
            the GCNConv keyword arguments

        Returns
        -------
        Tensor
            the output of the net
        """

        y = x
        for layer in self.net:
            y = layer(y, edge_index, **kwargs)
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

    def __getitem__(self, i):
        if isinstance(self.net[i], ResidualLayer):
            return self.net[i].layer[1]
        return self.net[i]