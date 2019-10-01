from .LossList import *


class GlobalLoss(LossList):
    """
    A class representing the global loss function.

    Attributes
    ----------
    alpha : float
        the weight of the loss (default is 1)
    inputFcn : callable
        a function to read correctly the input (default is nop)
    outputFcn : callable
        a function to read correctly the output (default is nop)
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
        super(GlobalLoss, self).__init__(*losses, alpha=1, name='Global', enabled=True, device=device)

    def eval(self, input, output):
        """
        Evaluate the global loss for the given network input and output

        Parameters
        ----------
        input : Data
            the given input to the network
        output : Data
            the produced output of the network

        Returns
        -------
        Tensor
            a single value Tensor representing the loss
        """

        self.value = torch.zeros(1, dtype=torch.float, device=self.device)
        if self.is_empty():
            warnings.warn('Global Loss cannot be evaluated while empty. The returned tensor carries no gradient.',
                          category=RuntimeWarning)
        else:
            for loss in self.loss:
                self.value += loss.eval(input, output)
        return self.value

    def to_dict(self, *args, **kwargs):
        """
        Convert the loss into a dictionary

        Returns
        -------
        dict
            A dictionary in the form {name : value}
        """

        return super(GlobalLoss, self).to_dict(compact=False)
