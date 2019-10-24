def is_frozen(model):
    """
    Returns True if the model is frozen, False otherwise

    Parameters
    ----------
    model : torch.nn.Module
        the model to check

    Returns
    -------
    bool
        True if the model is frozen, False otherwise
    """

    return not any([p.requires_grad for p in model.parameters()])
