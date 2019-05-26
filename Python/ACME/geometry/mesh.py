import torch
from utility.row         import *
from utility.col         import *
from utility.isempty     import *
from utility.unique      import *
from utility.accumarray  import *
from utility.FalseTensor import *
from utility.TrueTensor  import *
from topology.ispoly     import *
from topology.poly2edge  import *
from topology.poly2ind   import *
from .genus              import *

class Mesh(object):
    def __init__(self,Vertex=None,Normal=None,UV=None,Edge=None,Face=None,Hedra=None,name=''):
        self.Vertex = Vertex
        self.Normal = Normal
        self.UV     = UV
        self.Edge   = Edge
        self.Face   = Face
        self.Hedra  = Hedra
        self.name   = name
        if isempty(self.Edge):
            self.__compute_edge()
        self.__compute_external_face()
        self.__compute_external_vertex()
        self.__compute_external_edge()
        self.__compute_external_hedra()

    def nVertex(self):
        return row(self.Vertex)

    def nEdge(self):
        return col(self.Edge)

    def nFace(self):
        return col(self.Face)

    def nHedra(self):
        return col(self.Hedra)

    def isempty(self):
        return isempty(self.Vertex)

    def recompute_normals(self):
        self.Normal = vertex_normal(self.Vertex,self.Face)

    def isSurfaceMesh(self):
        return not isempty(self.Face)

    def isTriMesh(self):
        return self.isSurfaceMesh(self) and istri(self.Face)

    def isQuadMesh(self):
        return self.isSurfaceMesh(self) and isquad(self.Face)

    def isPolygonMesh(self):
        return self.isSurfaceMesh(self) and ispoly(self.Face)

    def isVolumetricMesh(self):
        return not isempty(self.Hedra)

    def isTetMesh(self):
        return self.isVolumetricMesh(self) and isquad(self.Hedra)

    def isHexMesh(self):
        return self.isVolumetricMesh(self) and ishex(self.Hedra)

    def isPolyhedralMesh(self):
        return self.isVolumetricMesh(self) and ispoly(self.Hedra)

    def genus(self):
        return genus(self.Vertex,self.Edge,self.Face,self.Hedra)

    def __compute_edge(self):
        if isempty(self.Face):
            return
        self.Edge = poly2edge(self.Face)[0]
        self.Edge = torch.t(unique(torch.t(torch.sort(self.Edge, dim=0)[0]),ByRows=True)[0])

    def __compute_external_vertex(self):
        self.ExternalVertex = FalseTensor(row(self.Vertex),device=self.Vertex.device)
        if not self.isVolumetricMesh():
            self.ExternalVertex = TrueTensor(row(self.Vertex),device=self.Vertex.device)
            return
        J,I = poly2lin(self.Face)
        self.ExternalVertex = accumarray(J,self.ExternalFace[I])>=1
        self.ExternalVertex = self.ExternalVertex

    def __compute_external_edge(self):
        self.ExternalEdge = FalseTensor(col(self.Edge),device=self.Vertex.device)
        if not self.isVolumetricMesh():
            self.ExternalEdge = TrueTensor(col(self.Edge),device=self.Vertex.device)
            return
        self.ExternalEdge = self.ExternalVertex(self.Edge[0,:]) and self.ExternalVertex(self.Edge[1,:])

    def __compute_external_face(self):
        self.ExternalFace = FalseTensor(col(self.Face),device=self.Vertex.device)
        if not self.isVolumetricMesh():
            self.ExternalFace = TrueTensor(col(self.Face),device=self.Vertex.device)
            return
        J = poly2lin(self.Hedra)[0]
        self.ExternalFace = accumarray(J,torch.ones_like(J))==1
        self.ExternalFace = self.ExternalFace

    def __compute_external_hedra(self):
        self.ExternalHedra = FalseTensor(col(self.Hedra),device=self.Vertex.device)
        if not self.isVolumetricMesh():
            return
        J,I = poly2ind(self.Hedra)
        self.ExternalHedra = accumarray(I,self.ExternalFace[J])>=1
        self.ExternalHedra = self.ExternalHedra
