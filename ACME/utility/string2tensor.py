import torch

def string2tensor(txt,prefix='',suffix='',separator=None,end='\n',dtype=torch.float,device='cuda:0'):
    """
    Converts a string into a Tensor.

    The function deletes from the string the given prefix and suffix, separates every row of the
    tensor using the given end character, and fetches the values using the given separator.

    Example:
        txt = '[1,2,3]\n[4,5,6]\n[7,8,9]'
        T   = string2tensor(txt,prefix='[',suffix=']',separator=',',end='\n',device='cpu')

    Parameters
    ----------
    txt : str
        the string to convert inot a tensor
    prefix : str (optional)
        the prefix of every entry in the string (default is '')
    suffix : str (optional)
        the suffix of every entry in the string (default is '')
    separator : str (optional)
        the separator used inbetween entries (default is None)
    end : str (optional)
        the character separating the tensor rows/cols (default is '\n')
    dtype : type (optional)
        the type of the output tensor (default is torch.float)
    device : str or torch.device
        the device where to store the tensor

    Returns
    -------
    Tensor
        A tensor continaing the valus in the string
    """

    token = txt.split(end)
    if not token[-1]:
        token = token[0:-1]
    token = [t.replace(prefix,'') for t in token]
    token = [t.replace(suffix,'') for t in token]
    token = [t.split(separator) for t in token]
    value = [[float(x) for x in t] for t in token]
    return torch.tensor(value,dtype=dtype,device=device)
