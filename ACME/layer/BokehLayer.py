import torch
from ..math.eye           import *
from ..topology.adjacency import *


class BokehLayer(torch.nn.Module):
    """
    A class representing a module for the bokeh effect

    Attributes
    ----------
    adj : Tensor
        the adjacency matrix of the view points

    Methods
    -------
    forward(x)
        returns the bokeh effect for the input tensor
    """

    def __init__(self, edge):
        """
        Parameters
        ----------
        edge : LongTensor
            the (2,N,) edge tensor of the view points topology
        """

        super(BokehLayer, self).__init__()
        self.adj = edge2adj(edge)
        self.adj = self.adj + eye_like(self.adj)
        self.adj = self.adj / torch.sum(self.adj, 1, keepdim=True)
        self.adj = self.adj.unsqueeze(0)

    def forward(self, x):
        """
        Returns the bokeh effect for the input tensor

        Parameters
        ----------
        x : Tensor
            the (B,N,...,) input tensor, where B stands for batch size

        Returns
        -------
        Tensor
            the transformed input tensor
        """

        x_hat = torch.matmul(self.adj, x.view(*x.size()[0:2], -1)).view(*x.size())
        return x_hat
