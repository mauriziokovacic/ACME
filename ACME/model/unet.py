from .autoencoder import *


class U_Net(AutoEncoder):
    """
    A class representing a U-Net

    Attributes
    ----------
    encoder : torch.nn.Module
        the encoder architecture
    decoder : torch.nn.Module
        the decoder architecture

    Methods
    -------
    forward(x)
        returns the U-Net output
    """

    def __init__(self, encoder, decoder, connection=None, name='U-Net', **kwargs):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            the encoder architecture. Must behave like a torch.nn.Sequential
        decoder : torch.nn.Module
            the decoder architecture. Must behave like a torch.nn.Sequential, containing HookLayers on the specified
            connection indices
        connection : LongTensor (optional)
            the (N,2,) indices tensor of the connected layers in encoders and decoders. If None if will be
            automatically computed (default is None)
        name : str (optional)
            the name of the U-Net (default is 'U-Net')
        """

        super(U_Net, self).__init__(encoder=encoder, decoder=decoder, name=name, **kwargs)
        if connection is None:
            connection = torch.cat((torch.arange(             0, len(encoder),  1).unsqueeze(1),
                                    torch.arange(len(decoder)-1,           -1, -1).unsqueeze(1),), dim=1)
        for i, j in connection:
            self.decoder[j].bind(self.encoder[i])

    def forward(self, x):
        """
        Returns the U-Net output

        Parameters
        ----------
        x : Tensor
            the U-Net input

        Returns
        -------
        Tensor
            the decoder output
        """

        return super(U_Net, self).forward(x)[0]
