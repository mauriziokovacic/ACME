def num_parameters(model):
    """
    Returns the number of parameters in the given model

    Parameters
    ----------
    model : torch.nn.Module
        the model to count the parameters of

    Returns
    -------
    int
        the number of parameters in the given model
    """

    n = 0
    for p in model.parameters():
        n += p.numel()
    return n
