from .FalseTensor import *
from .issparse    import *


def spy(self, size=(256, 256)):
    """
    Translates the input matrix into a binary matrix of the specified size, containing a 1 where there is a non-zero
    entry in the matrix

    Parameters
    ----------
    self : Tensor or SparseTensor
        the input matrix to spy
    size : tuple (optional)
        the output matrix size

    Returns
    -------
    Uint8Tensor
        the binary matrix
    """

    M = FalseTensor(*size, device=self.device)
    if issparse(self):
        i = self._indices().to(dtype=torch.float)
        i = (i / (torch.tensor(list(self.shape), dtype=torch.float).view(-1, 1) - 1))
        i = (i * (torch.tensor(list(M.shape), dtype=torch.float).view(-1, 1) - 1)).to(dtype=torch.long)
    else:
        i = torch.nonzero(self)
    M[tuple(i)] = 1
    return M
