def grad_flow(named_parameters):
    """
    Returns a dictionary containing the [min, mean, max] of each layer gradient

    Parameters
    ----------
    named_parameters : generator
        the generator returned by the named_parameters function of a model

    Returns
    -------
    dict
        a dictionary with the key being the layer name and its value being a [min, mean, max] list
    """

    d = {}
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            g = p.grad.abs()
            d[n] = [g.min().item(), g.mean().item(), g.max().item()]
    return d