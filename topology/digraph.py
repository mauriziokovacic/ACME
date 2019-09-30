import torch
from ..utility.row  import *
from ..utility.col  import *
from ..utility.find import *
from .adjacency     import *


class Digraph(object):
    """
    A class representing a directed graph

    Attributes
    ----------
    __adj__ : Tensor
        the adjacency matrix

    Methods
    -------
    successors(i)
        returns the successors indices of node i
    predecessors(i)
        returns the predecessors indices of node i
    roots()
        returns the root nodes indices
    leaves()
        returns the leaf nodes indices
    isroot(i)
        returns True if node i is a root, False otherwise
    isleaf(i)
        returns True if node i is a leaf, False otherwise
    isbranch(i)
        returns True if node i is a branch node, False otherwise
    isjoint(i)
        returns True if node i is a joint node, False otherwise
    size()
        returns the number of nodes in the graph
    isempty()
        returns True if the graph is empty
    from_adj(A)
        creates the grpah from the given adjacency matrix
    from_edges(E,W,size)
        creates the graph from the given edge tensor
    edges()
        returns the edge tensor
    matrix()
        returns the adjacency matrix
    add_nodes(n)
        adds n new nodes to the graph
    add_edges(E,W)
        adds the given edges to the graph
    """

    def __init__(self):
        self.__adj__ = None

    def from_edges(self, E, W=None, num_nodes=None):
        """
        Creates the graph from the given edge tensor

        Parameters
        ----------
        E : LongTensor
            the (2,N,) edge tensor
        W : Tensor (optional)
            the edge weights tensor. If None, all edges have weight 1 (default is 1)
        num_nodes : int (optional)
            the number of graph nodes. If None it will be automatically computed (default is None)

        Returns
        -------
        Digraph
            the graph
        """

        if W is None:
            W = torch.ones(col(E), dtype=torch.float, device=E.device)
        return self.from_adj(adjacency(E, W, size=num_nodes))

    def from_adj(self, A):
        """
        Creates the graph from the given adjacency matrix

        Parameters
        ----------
        A : Tensor
            the adjacency matrix

        Returns
        -------
        Digraph
            the graph
        """

        self.__adj__ = A
        return self

    def successors(self, i):
        """
        Returns the successors indices of node i

        Parameters
        ----------
        i : int
            the node index

        Returns
        -------
        LongTensor
            the successors indices
        """

        return find(self.__adj__[i, :] > 0)

    def predecessors(self, i):
        """
        Returns the predecessors indices of node i

        Parameters
        ----------
        i : int
            the node index

        Returns
        -------
        LongTensor
            the predecessors indices
        """

        return find(self.__adj__[:, i] > 0)

    def roots(self):
        """
        Returns the root nodes indices

        Returns
        -------
        LongTensor
            the roots indices
        """

        return find(torch.sum(self.__adj__, 0) == 0)

    def leaves(self):
        """
        Returns the leaf nodes indices

        Returns
        -------
        LongTensor
            the leaves indices
        """

        return find(torch.sum(self.__adj__, 1) == 0)

    def is_root(self, i):
        """
        Returns True if node i is a root, False otherwise

        Parameters
        ----------
        i : int
            the node index

        Returns
        -------
        bool
            True if node i is a root, False otherwise
        """

        return torch.sum(self.__adj__[:, i], 0) == 0

    def is_leaf(self, i):
        """
        Returns True if node i is a leaf, False otherwise

        Parameters
        ----------
        i : int
            the node index

        Returns
        -------
        bool
            True if node i is a leaf, False otherwise
        """

        return torch.sum(self.__adj__[i, :], 1) == 0

    def is_branch(self, i):
        """
        Returns True if node i is a branch node, False otherwise

        A branch node is a node which has more than one predecessors
        and/or more than one successors

        Parameters
        ----------
        i : int
            the node index

        Returns
        -------
        bool
            True if node i is a branch node, False otherwise
        """

        return torch.sum(self.__adj__[i, :], 1) > 1 or torch.sum(self.__adj__[:, i], 0) > 1

    def is_joint(self, i):
        """
        Returns True if node i is a joint node, False otherwise

        A joint node is a node which has one predecessor and one successor

        Parameters
        ----------
        i : int
            the node index

        Returns
        -------
        bool
            True if node i is a joint node, False otherwise
        """

        return torch.sum(self.__adj__[i, :], 1) == 1 * torch.sum(self.__adj__[:, i], 0) == 1

    def is_empty(self):
        """
        Returns True if the graph has no nodes, False otherwise

        Returns
        -------
        bool
            True if graph is empty, False otherwise
        """

        return self.size() == 0

    def size(self):
        """
        Returns the number of nodes in the graph

        Returns
        -------
        int
            the number of graph nodes
        """

        return row(self.__adj__)

    def edges(self):
        """
        Returns the edge tensor

        Returns
        -------
        LongTensor
            the edge tensor
        """

        return adj2edge(self.__adj__)

    def matrix(self):
        """
        Returns the adjacency matrix

        Returns
        -------
        LongTensor
            the edge tensor
        """

        return self.__adj__.clone()

    def add_nodes(self, n=1):
        """
        Adds n new nodes to the graph

        Parameters
        ----------
        n : int (optional)
            the number of new nodes

        Returns
        -------
        Digraph
            the graph
        """

        self.__adj__ = torch.cat((self.__adj__,
                                  torch.zeros(self.size(), n,   dtype=self.__adj__.dtype, device=self.__adj__.device)),
                                 dim=1)
        self.__adj__ = torch.cat((self.__adj__,
                                  torch.zeros(n, self.size()+n, dtype=self.__adj__.dtype, device=self.__adj__.device)),
                                 dim=0)
        return self

    def add_edge(self, E, W=None):
        """
        Adds the given edges to the graph

        Parameters
        ----------
        E : LongTensor
            the (2,N,) edge tensor
        W : Tensor (optional)
            the weight tensor. If None the weights are considered to be 1 (default is None)

        Returns
        -------
        Digraph
            the graph
        """

        if W is None:
            W = torch.ones(col(E), dtype=torch.float, device=E.device)
        for i, j, w in zip(*tuple(E), W):
            self.__adj__[i, j] = w
        return self

    def to(self, *args, **kwargs):
        """
        Calls the to method of all the contained tensors in the graph

        Returns
        -------
        Digraph
            the graph
        """

        if not self.isempty():
            self.__adj__.to(*args, **kwargs)
        return self

    def __repr__(self):
        return str(self.__adj__)
