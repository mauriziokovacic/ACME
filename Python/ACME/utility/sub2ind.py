import torch

def sub2ind(size,*tensors):
    """
    Converts subscripts to linear indices

    Parameters
    ----------
    size : list
        the size of the target tensor
    *tensors : LongTensor...
        a sequence of LongTensor containing the indices along a dimension

    Returns
    -------
    LongTensor
        the linear indices of the given subscripts
    """

    if len(tensors)==1:
        return tensors[0]
    T = torch.cat([make_column(t.flatten()) for t in tensors],dim=1)
    offset = torch.cumprod(torch.tensor(size[0:-1]),0).unsqueeze(0)
    offset = repmat(torch.cat((torch.ones(1,1,dtype=torch.long),offset),dim=1),row(T),1).to(device=T.device)
    return torch.sum(T*offset,1,keepdim=True)
