def triharmonic_rbf(x):
    """
    Computes the triharmonic radial basis function on the given input tensor

    Parameters
    ----------
    x : Tensor
        the (N,...) input distance tensor

    Returns
    -------
    Tensor
        the triharmonic radial basis function values
    """

    return x**3
