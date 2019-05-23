import torch

def tensor2string(tensor,prefix='',suffix='',separator=' ',end='\n'):
    """
    Converts a Tensor into a string.

    Parameters
    ----------
    tensor : Tensor
        an input Tensor
    prefix : str (optiona)
        the prefix to add before each row (default is '')
    suffix : str (optional)
        the suffix to add after each row (default is '')
    separator : str (optional)
        the separator to separate the values (default is ' ')
    end : str (optional)
        the character to end a row (default is '\n')

    Returns
    -------
    str
        a string representing the tensor in the given formats
    """

    if tensor is None:
        return ''
    tensor = tensor.cpu()
    txt = ('{}'+separator)*col(tensor)
    txt = (prefix+txt)[0:-len(separator)]
    txt = (txt+end)*row(tensor)
    txt = txt[0:-len(end)]
    return txt.format(*tuple([x.item() for x in tensor.flatten()]))
