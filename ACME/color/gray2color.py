def gray2color(C, dim=-1):
    """
    Converts the given grayscale colors into RGB format

    Parameters
    ----------
    C : Tensor
        the gray tensor
    dim : int (optional)
        the dimension along the operation is performed

    Returns
    -------
    Tensor
        the RGB grayscale tensor
    """

    c    = C.clone().unsqueeze(dim)
    size = c.shape
    size[dim] = 3
    return c.expand(*size)
