import torch
from ACME.utility.row      import *
from ACME.utility.col      import *
from ACME.utility.numel    import *
from ACME.utility.issquare import *
from ACME.utility.find     import *
from .adjacency            import *



class Digraph(object):
    """
    A class representing a directed graph

    Attributes
    ----------
    __adj : Tensor
        the adjacency matrix
    label : list
        a list of node names

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
    edges()
        returns the edge tensor
    """

    def __init__(E,W=None,size=None,label=None):
        """
        Parameters
        ----------
        E : LongTensor
            the (2,N,) edge tensor
        W : Tensor (optional)
            the edge weights tensor. If None, all edges have weight 1 (default is 1)
        size : int (optional)
            the number of graph nodes. If None it will be automatically computed (default is None)
        label : list (optional)
            the nodes names. If None they will be automatically set (default is None)
        """

        if issquare(E):
            self.__adj = E
        else:
            if W is None:
                W = torch.ones(col(E),dtype=torch.long,device=I.device)
            self.__adj = adjacency(E,W,size)
        if label is None:
            label = ['node{}'.format(i) for i in row(self.__adj)]
        self.label = label



    def successors(self,i):
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

        return find(self.__adj[i,:]>0)



    def predecessors(self,i):
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

        return find(self.__adj[:,i]>0)



    def roots(self):
        """
        Returns the root nodes indices

        Returns
        -------
        LongTensor
            the roots indices
        """

        return find(torch.sum(self.__adj,0)==0)



    def leaves(self):
        """
        Returns the leaf nodes indices

        Returns
        -------
        LongTensor
            the leaves indices
        """

        return find(torch.sum(self.__adj,1)==0)



    def isroot(self,i):
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

        return torch.sum(self.__adj[:,i],0)==0



    def isleaf(self,i):
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

        return torch.sum(self.__adj[i,:],1)==0



    def isbranch(self,i):
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

        return torch.sum(self.__adj[i,:],1)>1 or torch.sum(self.__adj[:,i],0)>1



    def isjoint(self,i):
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

        return torch.sum(self.__adj[i,:],1)==1 * torch.sum(self.__adj[:,i],0)==1



    def edges(self):
        """
        Returns the edge tensor

        Returns
        -------
        LongTensor
            the edge tensor
        """

        return torch.t(torch.nonzero(self.__adj))



    def __repr__(self):
        return self.__adj.__repr__()
