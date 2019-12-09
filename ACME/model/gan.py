from .model     import *


class GAN(Model):
    """
    A class representing a generic GAN model

    Attributes
    ----------
    G : torch.nn.Module
        the generator network
    D : torch.nn.Module
        the discriminator network
    name : str
        the name of the model
    device : str or torch.device
        the device to store the tensors to

    Methods
    -------
    forward(x, y)
        returns the discriminator results for a a pair of false and true tensors
    """

    def __init__(self, generator, discriminator, name='GAN', **kwargs):
        """
        Parameters
        ----------
        generator : torch.nn.Module
            the generator network
        discriminator : torch.nn.Module
            the discriminator network
        name : str (optional)
            the name of the module (default is 'GAN')
        kwargs : ...
            the other parameters of the Model class
        """

        super(GAN, self).__init__(name=name, **kwargs)
        self.add_module('G', generator)
        self.add_module('D', discriminator)

    def forward(self, x, y):
        """
        Returns the discriminator results for a a pair of false and true tensors

        Parameters
        ----------
        x : Tensor
            the noise tensor to be fed to the generator network
        y : Tensor
            the true tensor to be fed to the discriminator network

        Returns
        -------
        (Tensor, Tensor)
            the discriminator classification and the output of the generator
        """

        y_hat = self.G(x)
        c     = self.D(y, y_hat)
        return c, y_hat


class CycleGAN(Model):
    """
    A class representing a generic Cycle GAN model

    Attributes
    ----------
    F : GAN
        the forward GAN network
    G : GAN
        the backward GAN network
    name : str
        the name of the model
    device : str or torch.device
        the device to store the tensors to

    Methods
    -------
    forward(x, f, g)
        returns the network output
    """

    def __init__(self, F, G, name='CycleGAN', **kwargs):
        """
        Parameters
        ----------
        F : GAN
            the forward GAN network
        G : GAN
            the backward GAN network
        name : str (optional)
            the name of the model (default is 'CycleGAN')
        kwargs : ...
            the other parameters of the Model class
        """

        super(CycleGAN, self).__init__(name=name, **kwargs)
        self.add_module('F', F)
        self.add_module('G', G)

    def forward(self, x, f, g):
        """
        Returns the network output

        Parameters
        ----------
        x : Tensor
            the noise tensor to be fed to the forward generator network
        f : Tensor
            the true tensor to be fed to the forward discriminator network
        g : Tensor
            the true tensor to be fed to the backward discriminator network

        Returns
        -------
        (Tensor, Tensor, Tensor, Tensor)
            the forward and backward discriminators classification, and the generators outputs
        """

        c_f, g_hat = self.F(x, f)
        c_g, f_hat = self.G(g_hat, g)
        return c_f, c_g, g_hat, f_hat


### Alternative CycleGAN definition
# class CycleGAN(Model):
#     def __init__(self,
#                  generator,
#                  discriminator,
#                  generator_i,
#                  discriminator_i,
#                  name='CycleGAN', **kwargs):
#         super(CycleGAN, self).__init__(name=name, **kwargs)
#         F = GAN(generator  , discriminator  , name='ForwardGAN', **kwargs)
#         G = GAN(generator_i, discriminator_i, name='BackwardGAN', **kwargs)
#         self.add_module('F', self.F)
#         self.add_module('G', self.G)
#
#     def forward(self, x, f, g):
#         c_f, g_hat = self.F(x, f)
#         c_g, f_hat = self.G(g_hat, g)
#         return c_f, c_g, g_hat, f_hat
