def unfreeze(model):
    """
    Unfreezes all the parameters in the model

    Returns
    -------
    Model
        the model itself
    """

    for param in model.parameters():
        param.requires_grad = True
    model.training = True
    return model
