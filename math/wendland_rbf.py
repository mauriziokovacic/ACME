def Wendland_rbf(x, sigma=1):
    """
    Computes the Wendlad radial basis function for the given input distances

    Parameters
    ----------
    x : Tensor
        the (N,...) input distance tensor
    sigma : float (optional)
        the compact support of the function

    Returns
    -------
    Tensor
        the radial basis function values
    """

    return (1 - x / sigma) ** 4 + (4 * (x / sigma) + 1)
