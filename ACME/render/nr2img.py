from ACME.utility.torch2numpy import *

def nr2img(I):
    """
    Converts the output of the Neural Renderer into a Numpy RGB image

    Parameters
    ----------
    I : Tensor
        the output PyTorch tensor of Neural Renderer

    Returns
    -------
    Tensor
        a Numpy tensor
    """

    return torch2numpy(I.permute(3,2,1,0))[:,:,0:3,0]
