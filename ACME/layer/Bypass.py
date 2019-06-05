from torch.nn import Module

class Bypass(Module):
    """
    A layer performing no operation over the data

    Methods
    -------
    forward(input)
        performs no operation over the input
    """

    def __init__(self):
        super(Bypass,self).__init__(self)

    def forward(self,input):
        """
        Performs no operation over the input

        Parameters
        ----------
        input : Tensor
            the input batch tensor

        Returns
        -------
        Tensor
            the input batch tensor
        """

        return input
