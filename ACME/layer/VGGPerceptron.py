import torch


class VGGPerceptron(torch.nn.Module):
    """
    A class representing the standard VGG perceptron for feature extraction

    Attributes
    ----------
    model : torch.nn.Module
        the inner architecture

    Methods
    -------
    __create_model(cfg,in_channles,data_size,batch_norm)
        creates the inner architecture of this layer
    __create_layers(cfg,in_channels,batch_norm,dim)
        creates the inner convolutions layers
    forward(input)
        evaluates the inner architecture with the given input
    """

    def __init__(self,
             *data_size,
             cfg=[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
             in_channels=3,
             batch_norm=False,
            ):
        """
        Parameters
        ----------
        *data_size : tuple
            the data size in each dimension
        cfg : list (optional)
            a list containing the inner architecture configuration
        in_channels : int (optional)
            the number of input channels (default is 3)
        batch_norm : bool
            if True adds batch normalization layers (default is False)
        """

        super(VGGPerceptron, self).__init__()
        self.__create_model(cfg, in_channels, *data_size, batch_norm)

    def __create_model(self, cfg, in_channels, data_size, batch_norm):
        """
        Creates the inner architecture of this layer

        Parameters
        ----------
        cfg : list
            the inner architecture configuration
        in_channels : int
            the number of input channels
        data_size : tuple
            the data size in each dimension
        batch_norm : bool
            if True adds batch normalization layers

        Returns
        -------
        torch.nn.Module
            the inner architecture
        """

        layers, pool = self.__create_layers(cfg, in_channels=in_channels, batch_norm=batch_norm, dim=len(data_size))
        self.model = torch.nn.Sequential(
            *layers,
            eval('torch.nn.AdaptiveAvgPool'+str(len(data_size))+'d('+(str(tuple((eval(str(x)+'//'+str(2**pool))) for x in data_size)))+')')
        )

    def __create_layers(self, cfg, in_channels=3, batch_norm=False, dim=2):
        """
        Creates the inner convolutions layers

        Parameters
        ----------
        cfg : list
            the inner architecture configuration
        in_channels : int (optional)
            the number of input channels (default is 3)
        batch_norm : bool (optional)
            if True adds batch normalization layers (default is False)
        dim : int (optional)
            the input data number of dimensions (default is 2)

        Returns
        -------
        (list,int)
            the inner architecture layers in a list and the number of maxpooling
        """

        layers = []
        pool   = 0
        for v in cfg:
            if v == 'M':
                layers += [eval('torch.nn.MaxPool'+str(dim)+'d(kernel_size=2, stride=2)')]
                pool   += 1
            else:
                layers += [eval('torch.nn.Conv'+str(dim)+'d(in_channels, v, kernel_size=3, padding=1)')]
                if batch_norm:
                    layers += [eval('torch.nn.BatchNorm'+str(dim)+'d(v)')]
                layers += [torch.nn.ReLU(inplace=True)]
                in_channels = v
        return layers, pool

    def forward(self, input):
        """
        Evaluates the inner architecture with the given input

        Parameters
        ----------
        input : Tensor
            the input tensor

        Returns
        -------
        Tensor
            the extracted feature tensor
        """

        return self.model(input)
