import torch


class Extract_Attr(torch.nn.Module):
    """
    A layer extracting a specific attribute from the given input

    Attributes
    ----------
    attr : str
        the name of the attribute to extract

    Methods
    -------
    forward(input)
        returns the extracted attribute
    """

    def __init__(self, attr):
        super(Extract_Attr, self).__init__()
        """
        Parameters
        ----------
        attr : str
            the name of the attribute to extract
        """

        self.attr = attr

    def forward(self, input):
        """
        Returns the extracted attribute

        Parameters
        ----------
        input : object
            the input object

        Returns
        -------
        object
            the output attribute
        """

        return getattr(input, self.attr)
