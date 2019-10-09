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

    def __init__(self, dim=1):
        """
        Parameters
        ----------
        dim : int (optional)
            the dimension along the concatenation is performed (default is 1)
        """

        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, *inputs, **kwargs):
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

        return torch.cat(inputs, dim=self.dim)

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class Aggregation(Concatenate):
    """
    A layer performing a given operation over the concatenated inputs

    Attributes
    ----------
    dim : int
        the dimension along the operations are performed
    __aggregationFcn : callable
        the operation to be performed over the concatenated input

    Methods
    -------
    forward(*inputs, **kwargs)
        evaluates the aggregation function over the inputs
    """

    def __init__(self, aggregationFcn, dim=1):
        """
        Parameters
        ----------
        aggregationFcn : callable
            the operation to be performed over the concatenated input
        dim : int (optional)
            the dimension along the concatenation is performed (default is 1)
        """

        super(Aggregation, self).__init__(dim=dim)
        self.__aggregationFcn = aggregationFcn

    def forward(self, *inputs, **kwargs):
        """
        Evaluates the aggregation function over the inputs

        Parameters
        ----------
        inputs : Tensor...
            the input tensors
        kwargs : ...
            optional keyword arguments

        Returns
        -------
        Tensor
            the aggregated Tensor
        """
        return self.__aggregationFcn(
            super(Aggregation, self).forward(
                *tuple([i.unsqueeze(self.dim) for i in inputs]),
                **kwargs)
        )


class AddLayer(Aggregation):
    """
    A wrapper for an addition aggregation layer.
    It performs the sum of all the elements of the input tensor along the specified dimension.
    """

    def __init__(self, keepdim=False, dim=1):
        """
        Parameters
        ----------
        keepdim : bool (optional)
            if True keeps the tensor dimensions (default is True)
        dim : int (optional)
            the dimension along the concatenation is performed (default is 1)
        """

        super(AddLayer, self).__init__(lambda x: torch.sum(x, dim=dim, keepdim=keepdim), dim=dim)


class MeanLayer(Aggregation):
    """
    A wrapper for a mean aggregation layer.
    It performs the mean of all the elements of the input tensor along the specified dimension.
    """

    def __init__(self, keepdim=False, dim=1):
        """
        Parameters
        ----------
        keepdim : bool (optional)
            if True keeps the tensor dimensions (default is True)
        dim : int (optional)
            the dimension along the concatenation is performed (default is 1)
        """

        super(MeanLayer, self).__init__(lambda x: torch.mean(x, dim=dim, keepdim=keepdim), dim=dim)


class MinLayer(Aggregation):
    """
    A wrapper for a min aggregation layer.
    It performs the min of all the elements of the input tensor along the specified dimension.
    """

    def __init__(self, keepdim=False, dim=1):
        """
        Parameters
        ----------
        keepdim : bool (optional)
            if True keeps the tensor dimensions (default is True)
        dim : int (optional)
            the dimension along the concatenation is performed (default is 1)
        """

        super(MinLayer, self).__init__(lambda x: torch.min(x, dim=dim, keepdim=keepdim)[0], dim=dim)


class MaxLayer(Aggregation):
    """
    A wrapper for a max aggregation layer.
    It performs the max of all the elements of the input tensor along the specified dimension.
    """

    def __init__(self, keepdim=False, dim=1):
        """
        Parameters
        ----------
        keepdim : bool (optional)
            if True keeps the tensor dimensions (default is True)
        dim : int (optional)
            the dimension along the concatenation is performed (default is 1)
        """

        super(MaxLayer, self).__init__(lambda x: torch.max(x, dim=dim, keepdim=keepdim)[0], dim=dim)


class StdLayer(Aggregation):
    """
    A wrapper for a standard deviation aggregation layer.
    It performs the standard deviation of all the elements of the input tensor along the specified dimension.
    """

    def __init__(self, keepdim=False, dim=1):
        """
        Parameters
        ----------
        keepdim : bool (optional)
            if True keeps the tensor dimensions (default is True)
        dim : int (optional)
            the dimension along the concatenation is performed (default is 1)
        """

        super(StdLayer, self).__init__(lambda x: torch.std(x, dim=dim, keepdim=keepdim), dim=dim)


class HubLayer(Concatenate):
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

    def __init__(self, *models, dim=0):
        """
        Parameters
        ----------
        *models : torch.nn.Module...
            a sequence of modules
        dim : int (optional)
            the dimension along the concatenation is performed (default is 0)
        """

        super(HubLayer, self).__init__(dim=dim)
        self.add_module('models', torch.nn.ModuleList(*models))

    def forward(self, *inputs, **kwargs):
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

        if len(inputs) == 1:
            out = [f(*inputs, **kwargs) for f in self.models]
        else:
            out = [f(i, **kwargs) for f, i in zip(self.models, inputs)]
        return super(HubLayer, self).forward(*out)
