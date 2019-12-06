from .SparseTensor import *


def sparse_compact(self):
    """
    Compacts the input sparse tensor

    Parameters
    ----------
    self : SparseTensor
        a sparse tensor

    Returns
    -------
    SparseTensor
        a compact version of the input sparse tensor
    """

    i = self._indices()[:, self._values() != 0]
    v = self._values()[self._values() != 0]
    return SparseTensor(size=self.size, indices=i, values=v)