from ..utility.indices      import *
from ..utility.numel        import *
from ..utility.SparseTensor import *


def unitvec(size, i, sparse=False, dtype=torch.float, device='cuda:0'):
    """
    Creates a (|i|,size,) tensor with a single 1 in the specified positions

    Parameters
    ----------
    size : int
        number of elements in the tensor
    i : int
        position of the 1 in the tensor
    sparse : bool (optional)
        if True returns a sparse tensor, a dense tensor otherwise (default is False)
    dtype : type (optional)
        type of the tensor (default is torch.float)
    device : str or torch.device (optional)
        device where to store the tensor to (default is 'cuda:0')

    Returns
    -------
    Tensor
        a 1xn tensor
    """

    n = numel(i)
    j = indices(0, n-1, device=device).squeeze(-1)
    if sparse:
        e = SparseTensor(size=(n, size),
                         indices=torch.cat((j.unsqueeze(1), i.unsqueeze(1)), dim=1),
                         values=torch.ones(n, dtype=torch.float, device=device))
    else:
        e = torch.zeros(n, size, dtype=dtype, device=device)
        e[j, i] = 1
    return e
