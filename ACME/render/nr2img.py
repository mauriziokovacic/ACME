def nr2img(I):
    """
    Converts the output of the Neural Renderer into an RGB image

    Parameters
    ----------
    I : Tensor
        the (1,C,W,H,) tensor

    Returns
    -------
    Tensor
        a (W,H,C,) tensor
    """

    return I.permute(3,2,1,0)[:,:,0:3,0]
