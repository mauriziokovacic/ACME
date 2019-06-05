import torch

def polysort(T,winding=False):
    """
    Sorts the topology indices in ascending order

    Parameters
    ----------
    T : LongTensor
        the topology tensor
    winding : bool (optional)
        if True keeps winding ordering (default is False)

    Returns
    -------
    LongTensor
        the sorted topology
    """

    if winding:
        n = torch.argmin(T,0)
        return torch.t(torch.tensor([torch.roll(x,-k) for x,k in zip(torch.t(T),n)]))
    return torch.sort(T,0)[0]
