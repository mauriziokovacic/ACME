from .constant import *


class Range(object):
    """
    A class representing a continuous range of values

    Attributes
    ----------
    data : list
        a [min, max] pair

    Methods
    -------
    is_valid()
        returns True if the range is valid, False otherwise
    eval(value)
        returns True if the value is in range, False otherwise
    merge_(other)
        merges inplace two ranges together
    merge(other)
        merges two ranges together
    """

    def __init__(self, min=Inf, max=-Inf):
        """
        Parameters
        ----------
        inf : int or float
            the inferior of the range
        sup : int or float
            the superior of the range
        """
        self.data = [float(min), float(max)]

    def is_valid(self):
        """
        Returns True if the range is valid, False otherwise

        A range is valid when its inferior is lower or equal to its superior

        Returns
        -------
        bool
            if the range is valid
        """

        return self.data[0] <= self.data[1]

    def eval(self, value):
        """
        Returns True if the value is in range, False otherwise

        Parameters
        ----------
        value : int or float or Tensor
            the value to check

        Returns
        -------
        bool or Tensor
            True if the input is in range, False otherwise
        """

        return (value >= self.data[0]) * (value <= self.data[1])

    def merge_(self, other):
        """
        Merges inplace two ranges together

        Parameters
        ----------
        other : Range
            a range

        Returns
        -------
        Range
            the range itself
        """

        self.data[0] = min(self.data[0], other[0])
        self.data[1] = max(self.data[1], other[1])
        return self

    def merge(self, other):
        """
        Merges two ranges together

        Parameters
        ----------
        other : Range
            a range

        Returns
        -------
        Range
            a range
        """

        out = Range()
        out.merge_(self)
        out.merge_(other)
        return out

    def __eq__(self, other):
        self.data = other.data
        return self

    def __call__(self, value):
        return self.eval(value)

    def __contains__(self, value):
        return self.eval(value)

    def __getitem__(self, i):
        return self.data[i]

    def __repr__(self):
        return '{}'.format(self.data)
