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
            the dimension along the concatenation is performed (default is 0)
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



class AggregationLayer(Concatenate):
    """
    Aggregates the outputs of a sequence of models

    Attributes
    ----------
    models : torch.nn.Module...
        a sequence of modules

    Methods
    -------
    forward(*inputs)
        forwards the input(s) to each contained model
    """

    def __init__(self,*models,dim=0):
        """
        Parameters
        ----------
        *models : torch.nn.Module...
            a sequence of modules
        dim : int (optional)
            the dimension along the concatenation is performed (default is 0)
        """

        super(AggregationLayer,self).__init__()
        self.models = torch.nn.ModuleList(*models)



    def forward(self,*inputs):
        """
        Forwards the input(s) to each contained model

        Parameters
        ----------
        *inputs : object...
            a sequence of objects

        Returns
        -------
        Tensor
            the concatenated output of each model
        """

        if len(inputs)==1:
            out = [f(*inputs) for f in self.models]
        else:
            out = [f(i) for f,i in zip(self.models,inputs)]
        return super(AggregationLayer,self).forward(*out)
