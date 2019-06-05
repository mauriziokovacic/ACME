def index2label(index,label):
    """
    Converts an indices tensor into its labelled values counterpart

    Parameters
    ----------
    index : LongTensor
        the indices tensor
    label : list
        the labels values

    Returns
    -------
    list
        a list of labels
    """

    return [label[i] for i in index]
