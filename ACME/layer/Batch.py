import torch


class BatchFlatten(torch.nn.Module):
    """
    A layer performing the flattening of the input batch

    Methods
    -------
    forward(input)
        flattens the input batch
    """

    def __init__(self):
        super(BatchFlatten, self).__init__()

    def forward(self, input):
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


class BatchSum(torch.nn.Module):
    """
    A layer performing the sum of the input batch tensors

    Methods
    -------
    forward(input)
        the summed the input batch
    """

    def __init__(self):
        super(BatchSum, self).__init__()

    def forward(self, input):
        """
        Sums the input batch

        Parameters
        ----------
        input : Tensor
            the layer input batch

        Returns
        -------
        Tensor
            the sum of the batch tensor
        """

        return torch.sum(input, 0)


class BatchMean(torch.nn.Module):
    """
    A layer performing the mean of the input batch tensors

    Methods
    -------
    forward(input)
        the mean of the input batch
    """

    def __init__(self):
        super(BatchMean, self).__init__()

    def forward(self, input):
        """
        Returns the mean of the input batch

        Parameters
        ----------
        input : Tensor
            the layer input batch

        Returns
        -------
        Tensor
            the mean of the batch tensor
        """

        return torch.mean(input, 0)


class BatchReshape(torch.nn.Module):
    """
    A layer performing the reshape of the input batch

    Attributes
    ----------
    dim : tuple or list
        the new shape of the tensor

    Methods
    -------
    forward(input)
        reshapes the input batch
    """

    def __init__(self, dim):
        """
        Parameters
        ----------
        dim : tuple or list
            the new shape of the tensor
        """

        super(BatchReshape, self).__init__()
        self.dim = dim

    def forward(self, input):
        """
        Reshapes the input batch

        Parameters
        ----------
        input : Tensor
            the layer input batch

        Returns
        -------
        Tensor
            the reshaped batch
        """

        return torch.reshape(input, self.dim)


class BatchPermute(torch.nn.Module):
    """
    A layer performing the permutation of the input batch

    Attributes
    ----------
    dim : tuple or list
        the new shape of the tensor

    Methods
    -------
    forward(input)
        permutes the input batch
    """

    def __init__(self, *dim):
        """
        Parameters
        ----------
        dim : tuple or list
            the new dimension order of the tensor
        """

        super(BatchPermute, self).__init__()
        self.dim = dim

    def forward(self, input):
        """
        Permutes the input batch

        Parameters
        ----------
        input : Tensor
            the layer input batch

        Returns
        -------
        Tensor
            the permuted batch
        """

        return input.permute(self.dim)
