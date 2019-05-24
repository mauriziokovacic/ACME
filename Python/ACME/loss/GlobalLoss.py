import torch

class GlobalLoss(BaseLoss):
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



    def __init__(self,losses=None,device='cuda:0'):
        """
        Parameters
        ----------
        losses : list (optional)
            a list of losses to be evaluated (default is None)
        device : str (optional)
            a string indicating the device to use (default is 'cuda:0')
        """
        super(GlobalLoss,self).__init__(alpha=1,name='Global',enabled=True,device=device)
        self.reset()
        self.insert(losses)



    def eval(self,input,output):
        """
        Evaluate the loss for the given network input and output

        Parameters
        ----------
        input : Data
            the given input to the net
        output : Data
            the produced output of the net

        Returns
        -------
        Tensor
            a single value Tensor representing the loss
        """

        assert not self.empty(), "GlobalLoss cannot be evaluated while empty."
        self.value = torch.zeros(1,dtype=torch.float,device=self.device)
        for i in range(0,len(self.loss)):
            self.value += self.loss[i].eval(input,output)
        return self.value



    def size(self):
        """
        Returns the number of losses composing the global loss

        Returns
        -------
        int
            the length of the loss functions list.
        """

        return len(self)



    def empty(self):
        """
        Returns whether or not there are losses to be evaluated

        Returns
        -------
        bool
            True if the loss function list is empty, False otherwise
        """

        return not self.loss



    def insert(self,losses):
        """
        Add the given list of losses to the evaluation

        Parameters
        ----------
        losses : list
            A list of loss functions
        """

        if (losses is None) or (not losses):
           return
        self.loss += losses



    def reset(self):
        """
        Removes all contained losses and set the loss value to None
        """

        self.loss  = []
        self.value = None



    def to_dict(self):
        """
        Convert the loss into a dictionary

        Returns
        -------
        dict
            A dictionary in the form {name : value}
        """

        d = BaseLoss.to_dict(self)
        if not self.empty():
            for l in self.loss:
                d.update(l.to_dict())
        return d



    def __getitem__(self, i):
        """Returns the loss function at the i-th index"""
        return self.loss[i]



    def __len__(self):
        """Returns the number of losses contained in the global loss function"""
        if self.empty():
            return 0
        return len(self.loss)
