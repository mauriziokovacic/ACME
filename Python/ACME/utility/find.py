import torch

def find(cond,linear=True):
    """
    Finds the indices of the True values within the given condition

    Parameters
    ----------
    cond : uint8 Tensor
        a tensor derived from a condition (Ex.: t<0)
    linear : bool (optional)
        a flag driving the indices extraction. True returns the linear indices, False returns the subscripts (default is True)

    Returns
    -------
    Tensor
        a list of indices or a two-dimensional tensor containing the subscripts
    """

    if linear:
        i = indices(0,numel(cond)-1)
        cond = torch.flatten(cond)
        return i[cond]
    else:
        i = torch.meshgrid(*tuple(indices(0,d-1) for d in cond.shape))
        i = tuple(torch.flatten(np.transpose(x)) for x in i)
        cond = torch.flatten(cond)
        i = tuple(j[cond] for j in i)
    return i
