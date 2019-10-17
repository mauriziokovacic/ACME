import warnings
from ..utility.islist import *
from .Loss import *


class LossList(Loss):
    """
    A class representing the a series of loss functions.

    Attributes
    ----------
    alpha : float
        the weight of the loss (default is 1)
    inputFcn : callable
        a function to read correctly the input (default is nop)
    outputFcn : callable
        a function to read correctly the output (default is nop)
    name : str
        the loss name (default is 'List')
    enabled : bool
        a flag to enable or disable the loss computation (default is True)
    value : Tensor
        the last evaluated loss value (default is None)
    device : str
        a string indicating the device to use
    loss : list
        a list of the losses to be evaluated

    Methods
    -------
    size()
        returns the number of loss functions composing the loss
    is_empty()
        returns whether or not there are loss functions to be evaluated
    insert(*losses)
        inserts the given list of losses to the evaluation
    remove(i)
        removes the loss with the given index from the evaluation
    reset()
        Removes all contained loss functions and set the loss value to None
    to_dict()
        convert the loss into a dictionary {name:value}
    to(device)
        moves the loss (and its contained ones) to the given device
    """

    def __init__(self, *losses, name='List', **kwargs):
        """
        Parameters
        ----------
        losses : Loss...
            a sequence of losses to be evaluated
        name : str (optional)
            the name of the loss (default is 'List')
        **kwargs : keyword arguments...
            any keyword argument from Loss class
        """

        super(LossList, self).__init__(name=name, **kwargs)
        self.reset()
        self.insert(*losses)

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

        value = torch.mul(self.__eval__(input, output), self.alpha if self.enabled else 0)
        self.value = value.item()
        return value

    def __eval__(self, input, output):
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

        value = torch.zeros(1, dtype=torch.float, device=self.device)
        if self.is_empty():
            warnings.warn('LossList cannot be evaluated while empty. The returned tensor carries no gradient.',
                          category=RuntimeWarning)
        else:
            for loss in self.loss:
                value += loss.eval(input, output)
        return value

    def size(self):
        """
        Returns the number of losses composing the global loss

        Returns
        -------
        int
            the length of the loss functions list.
        """

        return len(self)

    def is_empty(self):
        """
        Returns whether or not there are losses to be evaluated

        Returns
        -------
        bool
            True if the loss function list is empty, False otherwise
        """

        return not self.loss

    def insert(self, *losses):
        """
        Inserts the given list of losses to the evaluation

        Parameters
        ----------
        losses : Loss...
            a sequence of loss functions

        Returns
        -------
        LossList
            the self object
        """

        if (losses is None) or (not losses):
            return self
        self.loss += losses if islist(losses) else list(losses)
        return self

    def remove(self, i):
        """
        Removes the i-th loss function from the list

        Parameters
        ----------
        i : int
            the index of the loss to be removed

        Returns
        -------
        LossList
            the self object

        Raises
        ------
        AssertionError
            if the given index is out of bounds
        """

        assert (i >= 0) and (i < len(self)), 'Index out of bounds'
        del self.loss[i]
        return self

    def reset(self):
        """
        Removes all contained losses and set the loss value to None
        """

        self.loss  = []
        self.value = None

    def to_dict(self, compact=True):
        """
        Convert the loss into a dictionary

        Parameters
        ----------
        compact : bool (optional)
            if True returns a one-entry dictionary, otherwise an entry for each contained loss (default is True)

        Returns
        -------
        dict
            A dictionary in the form {name : value}
        """

        d = Loss.to_dict(self)
        if not compact:
            if not self.is_empty():
                for l in self.loss:
                    d.update(l.to_dict())
        return d

    def to(self, device):
        """
        Moves the loss (and its contained ones) to the given device

        Parameters
        ----------
        device : str or torch.device
            the device to store the tensors to

        Returns
        -------
        Loss
            the loss itself
        """

        super(LossList, self).to(device)
        for loss in self.loss:
            loss.to(self.device)
        return self

    def __getitem__(self, i):
        """Returns the loss function at the i-th index"""
        return self.loss[i]

    def __len__(self):
        """Returns the number of losses contained in the global loss function"""
        return len(self.loss)
    
    def __repr__(self):
        text = '['
        for l in self.loss:
            text += l.__repr__()
        return text+']'
