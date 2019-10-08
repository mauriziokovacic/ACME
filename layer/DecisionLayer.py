import torch


class DecisionLayer(torch.nn.Module):
    """
    A fully connected layer performing a decision

    Attributes
    ----------
    model : torch.nn.Sequential
        the decision architecture

    Methods
    -------
    __create_model(input_size,output_size,bottleneck)
        creates the decision architecture
    forward(input)
        returns the decision tensor
    """

    def __init__(self, input_size, *cfg):
        """
        Parameters
        ----------
        input_size : int
            the input size
        cfg : tuple or list
            the output size of each fully connected layer
        """

        super(DecisionLayer, self).__init__()
        self.__create_layers(input_size, cfg)

    def __create_layers(self, input_size, cfg):
        """
        Creates the inner fully connected layers

        Parameters
        ----------
        input_size : int
            the number of input connections
        cfg : list
            the inner architecture configuration
        """

        layers = []
        for v in cfg[:-1]:
            layers += [torch.nn.Linear(input_size, v),
                       torch.nn.ReLU(inplace=True),
                       torch.nn.Dropout()]
            input_size = v
        layers += [torch.nn.Linear(input_size, cfg[-1])]
        self.model = torch.nn.Sequential(layers)

    def forward(self, input):
        """
        Returns the decision tensor

        Parameters
        ----------
        input : Tensor
            the input tensor

        Returns
        -------
        Tensor
            the decision tensor
        """

        return self.model(input)


def classifier_layer(input_size, *cfg):
    """
    Returns a standard classifier with the input architecture

    Parameters
    ----------
    input_size : int
        the input size
    cfg : tuple or list
        the output size of each fully connected layer

    Returns
    -------
    torch.nn.Module
        a standard classifier
    """

    return torch.nn.Sequential(
        DecisionLayer(input_size, *cfg),
        torch.nn.Softmax(),
    )


def VGG_classifier(input_size):
    """
    Returns the VGG classifier architecture

    Parameters
    ----------
    input_size : int
        the input size

    Returns
    -------
    torch.nn.Module
        the VGG classifier architecture
    """

    return classifier_layer(input_size, [4096, 4096, 1000])
