def freeze(model):
    """
    Freezes all the parameters in the model

    Returns
    -------
    Model
        the model itself
    """

    for param in model.parameters():
        param.requires_grad = False
    model.training = False
    return model
