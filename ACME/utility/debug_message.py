def debug_message(message, verbose, no_newline=False):
    """
    Prints the given message if the verbose flag is true.

    Parameters
    ----------
    message : str
        The message to be printed
    verbose : bool
        The flag enabling the printing
    no_newline : bool (optional)
        If True does not end the message with the newline character
    """

    if verbose:
        print(message, end='' if no_newline else '\n')
