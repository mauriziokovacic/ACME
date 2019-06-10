import torch

class MLPLayer(torch.nn.Module):
    """
    A class representing a MLP layer

    Attributes
    ----------
    model : torch.nn.Module
        the inner architecture

    Methods
    -------
    __create_model(cfg,in_channles,data_size,batch_norm)
        creates the inner architecture of this layer
    forward(input)
        evaluates the inner architecture with the given input
    """

    def __init__(self,
             *data_size,
             cfg=[32,32],
             in_channels=3,
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

        super(MLPLayer,self).__init__()
        self.model = self.__create_model(cfg,in_channels,*data_size)



    def __create_model(self,cfg,in_channels,data_size):
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

        Returns
        -------
        torch.nn.Module
            the inner architecture
        """

        layers = []
        for v in cfg:
                layers += [eval('torch.nn.Conv'+str(len(data_size))+'d(in_channels, v, kernel_size=1, padding=1)')]
                layers += [eval('torch.nn.BatchNorm'+str(len(data_size))+'d(v)')]
                layers += [torch.nn.ReLU(inplace=True)]
                in_channels = v
        return torch.nn.Sequential(*layers)



    def forward(self,input):
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
