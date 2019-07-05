import torch


class Batch_Flatten(torch.nn.Module):
    """
    A layer performing the flattening of the input batch

    Methods
    -------
    forward(input)
        flattens the input batch
    """

    def __init__(self):
        super(Batch_Flatten, self).__init__()

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


class Batch_Sum(torch.nn.Module):
    """
    A layer performing the sum of the input batch tensors

    Methods
    -------
    forward(input)
        the summed the input batch
    """

    def __init__(self):
        super(Batch_Sum, self).__init__()

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


class Batch_Mean(torch.nn.Module):
    """
    A layer performing the mean of the input batch tensors

    Methods
    -------
    forward(input)
        the mean of the input batch
    """

    def __init__(self):
        super(Batch_Mean, self).__init__()

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


class Batch_Reshape(torch.nn.Module):
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

        super(Batch_Reshape, self).__init__()
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


class Batch_Permute(torch.nn.Module):
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

        super(Permute_Channel, self).__init__()
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
