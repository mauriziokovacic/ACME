import torch
from ..utility.ACMEClass import *
from ..utility.identity  import *


class Loss(ACMEClass):
    """
    A class representing the base for any loss function.

    Attributes
    ----------
    alpha : float
        the weight of the loss
    inputFcn : callable
        a function to read correctly the input (default is identity)
    outputFcn : callable
        a function to read correctly the output (default is identity)
    name : str
        the loss name
    enabled : bool
        a flag to enable or disable the loss computation
    value : Tensor
        the last evaluated loss value (default is None)
    device : str
        a string indicating the device to use

    Methods
    -------
    eval(input,output)
        Evaluate the loss for the given network input and output
    toggle(status)
        Toggle the loss evaluation on or off depending on status
    enable()
        Enable the loss evaluation. Equivalent to toggle(True).
    disable()
        Disable the loss evaluation. Equivalent to toggle(False).
    is_enabled()
        Returns whether the loss evaluation is enabled or not.
    to_dict()
        Convert the loss into a dictionary entry {name : value}
    to(device)
        moves the loss to the given device
    """

    def __init__(self, alpha=1, inputFcn=identity, outputFcn=identity, name='', enabled=True, device='cuda:0'):
        """
        Parameters
        ----------
        alpha : float (optional)
            the weight of the loss (default is 1)
        name : str (optional)
            the loss name (default is '')
        enabled : bool (optional)
            a flag to enable or disable the loss computation (default is True)
        device : str (optional)
            a string indicating the device to use (default is 'cuda:0')
        """

        self.alpha     = alpha
        self.inputFcn  = inputFcn
        self.outputFcn = outputFcn
        self.name      = name
        self.enabled   = enabled
        self.value     = None
        self.device    = device

    def eval(self, input, output):
        """
        Evaluate the loss for the given network input and output

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

        value = torch.mul(self.__eval__(self.inputFcn(input), self.outputFcn(output)),
                          self.alpha if self.enabled else 0)
        self.value = value.item()
        return value

    def __eval__(self, input, output):
        """
        Interface for computing the loss.

        Parameters
        ----------
        input : Data
            the given input to the net
        output : Data
            the produced output of the net

        Raises
        ------
        NotImplementedError
            if derived class does not implement this method
        """

        raise NotImplementedError

    def toggle(self, status):
        """
        Toggle the loss evaluation on or off depending on status

        Parameters
        ----------
        status : bool
            Status of the loss function.
        """

        self.enabled = status

    def enable(self):
        """Enable the loss evaluation. Equivalent to toggle(True)."""

        self.toggle(True)

    def disable(self):
        """Disable the loss evaluation. Equivalent to toggle(False)."""

        self.toggle(False)

    def is_enabled(self):
        """
        Returns whether the loss evaluation is enabled or not.

        Returns
        -------
        bool
            True if the loss evaluation is enabled, False otherwise.
        """

        return self.enabled

    def to_dict(self, *args, **kwargs):
        """
        Convert the loss into a dictionary entry {name : value}

        Returns
        -------
        dict
            A single entry dictionary in the form {name : value}
        """

        return self.cast(dict)

    def __dict__(self):
        return {self.name: self.value}

    def to(self, device):
        """
        Moves the loss to the given device

        Parameters
        ----------
        device : str or torch.device
            the device to store the tensors to

        Returns
        -------
        Loss
            the loss itself
        """

        self.device = device
        return self

    def __gt__(self, other):
        """Returns True if the current loss function evaluates more than the other, False otherwise."""

        return self.value > other.value

    def __lt__(self, other):
        """Returns True if the current loss function evaluates less than the other, False otherwise."""

        return self.value < other.value

    def __eq__(self, other):
        """Returns True if the current loss function evaluates equal to the other, False otherwise."""

        return self.value == other.value

    def __repr__(self):
        return str(self.name) + " Loss"

    def __call__(self, input, output):
        return self.eval(input, output)
