def mvs2texture(I):
    """
    Converts a Multi View Stack tensor into a 3D texture

    Parameters
    ----------
    I : Tensor
        a Multi View Stack tensor

    Returns
    -------
    Tensor
        a 3D texture tensor
    """

    return I.permute(1, 0, 2, 3)
