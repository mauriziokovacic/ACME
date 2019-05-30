import torch

class Concatenate(torch.nn.Module):
    """
    A layer concatenating the inputs along the specified dimension

    Attributes
    ----------
    dim : int
        the dimension along the concatenation is performed

    Methods
    -------
    forward(*input)
        concatenates the inputs
    """

    def __init__(self,dim=0):
        """
        Parameters
        ----------
        dim : int (optional)
            the dimension along the concatenation is performed
        """

        super(Concatenate,self).__init__()
        self.dim = dim



    def forward(self,*inputs):
        """
        Concatenates the input

        Parameters
        ----------
        *inputs : Tensor...
            a sequence of Tensors

        Returns
        -------
        Tensor
            the concatenated tensors
        """

        return torch.cat(*inputs,dim=self.dim)
