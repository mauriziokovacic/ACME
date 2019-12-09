from ..geometry.subdivide import *


class MeshUnpooling(torch.nn.Module):
    """
    A class representing a mesh unpooling layer.

    Attributes
    ----------
    cached : bool
        if True caches the pooling data, otherwise computes it at every input
    matrix : Tensor
        the pooling matrix
    face : LongTensor
        the topology tensor

    Methods
    -------
    forward(x, *args)
        unpools the input data
    """

    def __init__(self, matrix=None, face=None, cached=True):
        """
        Parameters
        ----------
        matrix : Tensor (optional)
            the unpool matrix (default is None)
        face : LongTensor (optional)
            the topology tensor (default is None)
        cached : bool (optional)
            if True caches the unpooling data, otherwise computes it at every input (default is True)
        """

        super(MeshUnpooling, self).__init__()
        self.cached = cached
        self.matrix = matrix
        self.face   = face

    def forward(self, x, *args):
        """
        Unpools the input data

        Parameters
        ----------
        x : Data
            the input mesh with N vertices
        args : Tensor...
            optional (N,D,) tensors

        Returns
        -------
        Data or (Data, Tensor...)
            the unpooled mesh and the other input data
        """

        if self.matrix is None or not self.cached:
            self.face, self.matrix = subdivide(x.pos, x.face)[1:]
            self.matrix.requires_grad = False
        x.pos  = torch.matmul(self.matrix, x.pos)
        x.norm = torch.matmul(self.matrix, x.norm)
        x.face = self.face
        if len(args) == 0:
            return x
        return (x, ) + tuple([torch.matmul(self.matrix, a) for a in args])
