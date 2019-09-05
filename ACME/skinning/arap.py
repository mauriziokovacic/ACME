from ACME.math import *
from ACME.geometry import *

def arap_energy(P, P_prime, R, wij, E):
    """
    Computes the ARAP energy as shown in Alexa and Sorkine paper.
    
    Parameters
    ----------
    P : Tensor
        the (N,3,) points set tensor in undeformed state
    P_prime : Tensor
        a (N,3,) points set tensor in deformed state
    R : Tensor
        the (N,3,3,) rotation matrices tensor, described in the article
    wij : Tensor
        the (E,1,) tensor containing the weights for the mesh edges
    E : LongTensor
        the (2,E,) edge tensor
        
    Returns
    -------
    Tensor
        a (1,) tensor representing the ARAP energy
    """
    
    Pi_prime, Pj_prime = P_prime[E]
    Pi, Pj = P[E]
    Ri = R[E[0]]
    return torch.sum(wij * sqnorm((Pi_prime - Pj_prime) - torch.matmul(Ri, torch.t(Pi - Pj).unsqueeze(2).permute(1, 0, 2)).squeeze(), 1))


def arap_S(Eij, Dij, Eij_prime):
    """
    Computes the ARAP S covariance matrix for point Pi as described in the article.
    
    Parameters
    ----------
    Eij : Tensor
        the (N,3,) tensor represeting the vector Pi-Pj in the undeformed state
    Dij : Tensor
        a (N,N,) diagonal matrix with the edge weights
    Eij_prime : Tensor
        the (N,3,) tensor represeting the vector Pi-Pj in the deformed state
        
    Returns
    -------
    Tensor
        a (3,3,) tensor representing the covariance matrix of point Pi
    """
    
    return torch.matmul(torch.t(Eij), torch.matmul(Dij, Eij_prime))


def arap_R(Si):
    """
    Given the covariance matrix Si, computes the ARAP rotation for point Pi
    
    Parameters
    ----------
    Si : Tensor
        the (3,3,) tensor represeting the covariance matrix of point Pi
        
    Returns
    -------
    Tensor
        a (3,3,) tensor representing the ARAP rotation matrix of point Pi
    """
    
    U, _, V = torch.svd(Si)
    return torch.matmul(V, torch.t(U))


def arap_Si(P, P_prime, A, i):
    """
    Computes the ARAP S covariance matrix for the i-th as described in the article.
    
    Parameters
    ----------
    P : Tensor
        the (N,3,) points set tensor in undeformed state
    P_prime : Tensor
        a (N,3,) points set tensor in deformed state
    A : Tensor
        the (N,N,) adjacency matrix computed from the input mesh
    i : int
        the point index the covariance matrix has to be computed for
        
    Returns
    -------
    Tensor
        a (3,3,) tensor representing the covariance matrix of point Pi
    """
    
    j         = find(A[i, :])
    Eij       = P[i, :] - P[j, :]
    Eij_prime = P_prime[i, :] - P_prime[j, :]
    Dij       = diag(A[i, j])
    return arap_S(Eij, Dij, Eij_prime)


def arap_metric(P, P_prime, A, E):
    """
    Computes the ARAP energy of the input model
    
    Parameters
    ----------
    P : Tensor
        the (N,3,) points set tensor in undeformed state
    P_prime : Tensor
        a (N,3,) points set tensor in deformed state
    A : Tensor
        the (N,N,) adjacency matrix computed from the input mesh
    E : LongTensor
        the (2,E,) edge tensor
        
    Returns
    -------
    Tensor
        a (1,) tensor representing the ARAP energy
    """
    
    R = torch.cat([arap_R(arap_Si(P, P_prime, A, i)).unsqueeze(0) for i in range(row(P))], dim=0)
    return arap_energy(P, P_prime, R, A[tuple(E)], E)
