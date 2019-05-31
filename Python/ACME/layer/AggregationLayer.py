import torch

class AggregationLayer(torch.nn.Module):
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

    def __init__(self,*models):
        """
        Parameters
        ----------
        *models : torch.nn.Module...
            a sequence of modules
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
        return torch.cat(out)
