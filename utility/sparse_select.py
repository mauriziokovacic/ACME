from .find         import *
from .indices      import *
from .numel        import *
from .SparseTensor import *


def sparse_select(self, *index, dim=0):
    """
    Selects the values at the given indices along the specified dimension

    Parameters
    ----------
    self : SparseTensor
        the input sparse tensor
    index : LongTensor
        the values indices to extract
    dim : int (optional)
        the dimension to select the values from

    Returns
    -------
    SparseTensor
        the selected sparse tensor values
    """

    j = torch.cat([find(self._indices()[dim, :] == i) for i in index])
    i = self._indices()[:, j].t()
    i[:, dim] = indices(0, numel(j)-1, device=i.device).squeeze()
    s = list(self.shape)
    s[dim] = numel(j)
    return SparseTensor(size=s, indices=i, values=self._values()[j])