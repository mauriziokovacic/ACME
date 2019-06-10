from ACME.utility.LongTensor import *

def label2index(value,label,device='cuda:0'):
    """
    Converts a given sequence of labelled values into a indices tensor

    Parameters
    ----------
    value : list
        a sequence of labelled values
    label : list
        the label set
    device : str or torch.device (optional)
        the device the tensor will be stored to (default is 'cuda:0')

    Returns
    -------
    LongTensor
        the indices tensor
    """

    d = dict(zip(label,list(range(0,len(label)))))
    return LongTensor([d[x] for x in *value],device=device)
