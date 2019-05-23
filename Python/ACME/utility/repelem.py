import numpy as np
import torch

def repelem(tensor,*size):
    """
    Repeats the tensor values along the tensor dimensions by the given times

    Example:
        repelem([[1,2,3]],1,2) -> [[1,1,2,2,3,3]]

    Parameters
    ----------
    tensor : Tensor
        the input tensor
    *size : int...
        a sequence of times to repeats the tensor values along a particular dimension

    Returns
    -------
    Tensor
        a tensor
    """

    out = tensor.cpu().numpy()
    for d in range(0,len(size)):
        out = np.repeat(out,size[d],axis=d)
    return torch.from_numpy(out).to(dtype=tensor.dtype,device=tensor.device)
