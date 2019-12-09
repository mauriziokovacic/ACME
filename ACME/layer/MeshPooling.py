from ..geometry.unsubdivide import *


class MeshPooling(torch.nn.Module):
    """
    A class representing a mesh pooling layer. It supposes the input mesh is trivially poolable

    Attributes
    ----------
    cached : bool
        if True caches the pooling data, otherwise computes it at every input
    index : LongTensor
        the vertices indices
    face : LongTensor
        the topology tensor

    Methods
    -------
    forward(x, *args)
        pools the input data
    """

    def __init__(self, index=None, face=None, cached=True):
        """
        Parameters
        ----------
        index : LongTensor (optional)
            the vertices indices (default is None)
        face : LongTensor (optional)
            the topology tensor (default is None)
        cached : bool (optional)
            if True caches the pooling data, otherwise computes it at every input (default is True)
        """

        super(MeshPooling, self).__init__()
        self.cached = cached
        self.index  = index
        self.face   = face

    def forward(self, x, *args):
        """
        Pools the input data

        Parameters
        ----------
        x : Data
            the input mesh with N vertices
        args : Tensor...
            optional (N,D,) tensors

        Returns
        -------
        Data or (Data, Tensor...)
            the pooled mesh and the other input data
        """

        if self.matrix is None or not self.cached:
            self.face, self.index = unsubdivide(x.pos, x.face)[1:]
        x.pos  = x.pos[self.index]
        x.norm = x.norm[self.index]
        x.face = self.face
        if len(args) == 0:
            return x
        return (x, ) + tuple([a[self.index] for a in args])
