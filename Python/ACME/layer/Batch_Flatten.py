from torch.nn import Module

class Batch_Flatten(Module):
    """
    A layer performing the flattening of the input batch

    Methods
    -------
    forward(input)
        flattens the input batch
    """

    def __init__(self):
        super(Batch_Flatten,self).__init__()



    def forward(self,input):
        """
        Flattens the input batch

        Parameters
        ----------
        input : Tensor
            the layer input batch

        Returns
        -------
        Tensor
            the flattened batch
        """

        return input.flatten()
