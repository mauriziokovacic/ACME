def weight_init(model, value=0):
    """
    Initializes all the given model parameters to the specified value

    Parameters
    ----------
    model : torch.nn.Module
        the model to set the parameters for
    value : int or float (optional)
        the value to set the parameters to (default is 0)

    Returns
    -------
    torch.nn.Module
        the model itself
    """

    for p in model.parameters():
        p.data.fill_(value)
    return model
