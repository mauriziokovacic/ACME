import torch
from .LossList import *


class GlobalLoss(LossList):
    """
    A class representing the global loss function.

    Attributes
    ----------
    alpha : float
        the weight of the loss (default is 1)
    name : str
        the loss name (default is 'Global')
    enabled : bool
        a flag to enable or disable the loss computation (default is True)
    value : Tensor
        the last evaluated loss value (default is None)
    device : str
        a string indicating the device to use
    loss : list
        a list of the losses to be evaluated
    compact : bool
        if True returns a single entry dict, else lists every contained loss

    Methods
    -------
    size()
        Returns the number of losses composing the global loss
    empty()
        Returns whether or not there are losses to be evaluated
    insert(losses)
        Add the given list of losses to the evaluation
    reset()
        Removes all contained losses and set the loss value to None
    to_dict()
        Convert the loss into a dictionary {name:value}
    """

    def __init__(self, *losses, device='cuda:0'):
        """
        Parameters
        ----------
        losses : Loss...
            a sequence of loss functions to be evaluated
        device : str or torch.device (optional)
            the device the tensors will be stored to (default is 'cuda:0')
        """
        super().__init__(*losses, alpha=1, name='Global', enabled=True, device=device, compact=False)
