from utility.flipud

def polyflip(T):
    """
    Flips the polygons in the topology

    Parameters
    ----------
    T : Tensor
        the topology tensor

    Returns
    -------
    Tensor
        the flipped topology tensor
    """

    return flipud(T)
