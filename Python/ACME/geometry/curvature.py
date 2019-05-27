import torch
from utility.accumarray import *
from math.norm          import *
from math.angle         import *
from .area              import *



def mean_curvature_normal(P,T):
    A = torch.reciprocal(torch.mul(barycentric_area(P,T),2))
    L = cotangent_Laplacian(P,T)
    return A * torch.mm(L,P)



def mean_curvature(P,T,dense=True):
    Hn = mean_curvature_normal(P,T)
    return torch.mul(norm(Hn),0.5)



def gaussian_curvature(P,T):
    Pi,Pj,Pk = P[T]
    Eij   = normr(Pj-Pi)
    Ejk   = normr(Pk-Pj)
    Eki   = normr(Pi-Pk)
    A     = barycentric_area(P,T)
    theta = accumarray(torch.cat(tuple(T)).squeeze(),torch.cat((angle(Eij,-Eki),angle(Ejk,-Eij),angle(Eki,-Ejk))).squeeze())
    return (PI2-to_column(theta)) / A



def max_curvature(P,T):
    H = mean_curvature(P,T)
    K = gaussian_curvature(P,T)
    H2K = torch.pow(H,2)-K
    return H+torch.sqrt(torch.where(H2K>0,H2K,torch.zeros_like(H)))



def min_curvature(P,T):
    H   = mean_curvature(P,T)
    K   = gaussian_curvature(P,T)
    H2K = torch.pow(H,2)-K
    return H-torch.sqrt(torch.where(H2K>0,H2K,torch.zeros_like(H)))
