def assertion(cond, msg=''):
    """
    Asserts if condition is false and display a message

    Parameters
    ----------
    cond : bool
        a condition that must evaluate true
    msg : str (optional)
        a message to be displayed if assert fails (default is '')
    """

    assert cond, msg
