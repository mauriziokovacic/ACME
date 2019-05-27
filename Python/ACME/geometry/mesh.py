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
    """
    A class representing a generic mesh.

    Attributes
    ----------
    Vertex : Tensor
        the mesh vertices
    Normal : Tensor
        the mesh vertex normals
    UV : Tensor
        the mesh vertex UVs
    Edge : LongTensor
        the mesh edges
    Face : LongTensor
        the mesh faces
    Hedra : LongTensor
        the mesh volumes
    name : str
        the mesh name
    ExternalVertex : Uint8Tensor
        the external vertices flag
    ExternalEdge : Uint8Tensor
        the external edges flag
    ExternalFace : Uint8Tensor
        the external faces flag
    ExternalHedra: Uint8Tensor
        the external volumes flag

    Methods
    -------
    nVertex()
        Returns the number of vertices in the mesh
    nEdge()
        Returns the number of edges in the mesh
    nFace()
        Returns the number of faces in the mesh
    nHedra()
        Returns the number of volumes in the mesh
    isempty()
        Returns wheter or not the mesh is empty
    recompute_normals()
        Recomputes the vertex normals
    isSurfaceMesh()
        Returns True if the mesh has a surface
    isTriMesh()
        Returns True if the mesh is a triangle mesh
    isQuadMesh()
        Returns True if the mesh is a quad mesh
    isPolygonMesh()
        Returns True if the mesh is a generic polygonal mesh
    isVolumetric()
        Returns True if the mesh has volumes
    isTetMesh()
        Returns True if the mesh is a tetrahedral mesh
    isHexMesh()
        Returns True if the mesh is an hexaedral mesh
    isPolyhedralMesh()
        Returns True if the mesh is a generic polyhedral mesh
    genus()
        Returns the genus of the mesh
    update_ext()
        Updates the external flags for vertices, edges, faces and volumes
    __compute_edge()
        Computes the mesh edges
    __compute_external_vertex()
        Computes the external vertices flags
    __compute_external_edge()
        Computes the external edges flags
    __compute_external_face()
        Computes the external faces flags
    __compute_external_hedra()
        Computes the external volumes flags
    """



    def __init__(self,Vertex=None,Normal=None,UV=None,Edge=None,Face=None,Hedra=None,name=''):
        """
        Parameters
        ----------
        Vertex : Tensor (optional)
            the mesh vertices (default is None)
        Normal : Tensor (optional)
            the mesh vertex normals (default is None)
        UV     : Tensor (optional)
            the mesh vertex UVs (default is None)
        Edge   : LongTensor (optional)
            the mesh edges. If None they will be automatically computed (default is None)
        Face   : LongTensor (optional)
            the mesh faces (default is None)
        Hedra  : LongTensor (optional)
            the mesh volumes (default is None)
        name   : str (optional)
            the mesh name (default is '')
        """

        self.Vertex = Vertex
        self.Normal = Normal
        self.UV     = UV
        self.Edge   = Edge
        self.Face   = Face
        self.Hedra  = Hedra
        self.name   = name
        if isempty(self.Edge):
            self.__compute_edge()
        self.update_ext()



    def nVertex(self):
        """
        Returns the number of vertices in the mesh

        Returns
        -------
        int
            the number of vertices
        """

        return row(self.Vertex)



    def nEdge(self):
        """
        Returns the number of edges in the mesh

        Returns
        -------
        int
            the number of edges
        """

        return col(self.Edge)



    def nFace(self):
        """
        Returns the number of faces in the mesh

        Returns
        -------
        int
            the number of faces
        """

        return col(self.Face)



    def nHedra(self):
        """
        Returns the number of volumes in the mesh

        Returns
        -------
        int
            the number of volumes
        """

        return col(self.Hedra)



    def isempty(self):
        """
        Returns wheter or not the mesh is empty

        Returns
        -------
        bool
            True if the mesh has no vertices, False otherwise
        """

        return isempty(self.Vertex)



    def recompute_normals(self):
        """Recomputes the vertex normals"""

        self.Normal = vertex_normal(self.Vertex,self.Face)



    def isSurfaceMesh(self):
        """
        Returns whether the mesh has a surface or not

        Returns
        -------
        bool
            True if the mesh has faces, False otherwise
        """

        return not isempty(self.Face)



    def isTriMesh(self):
        """
        Returns whether the mesh is a triangle mesh or not

        Returns
        -------
        bool
            True if the mesh has triangle faces, False otherwise
        """

        return self.isSurfaceMesh(self) and istri(self.Face)



    def isQuadMesh(self):
        """
        Returns whether the mesh is a quad mesh or not

        Returns
        -------
        bool
            True if the mesh has quad faces, False otherwise
        """

        return self.isSurfaceMesh(self) and isquad(self.Face)



    def isPolygonMesh(self):
        """
        Returns whether the mesh is a polygonal mesh or not

        Returns
        -------
        bool
            True if the mesh has polygonal faces, False otherwise
        """

        return self.isSurfaceMesh(self) and ispoly(self.Face)



    def isVolumetricMesh(self):
        """
        Returns whether the mesh has volumes or not

        Returns
        -------
        bool
            True if the mesh has volumes, False otherwise
        """

        return not isempty(self.Hedra)



    def isTetMesh(self):
        """
        Returns whether the mesh is a tetrahedral mesh or not

        Returns
        -------
        bool
            True if the mesh has tetrahedral volumes, False otherwise
        """

        return self.isVolumetricMesh(self) and isquad(self.Hedra)



    def isHexMesh(self):
        """
        Returns whether the mesh is a haxaedral mesh or not

        Returns
        -------
        bool
            True if the mesh has hexaedral volumes, False otherwise
        """

        return self.isVolumetricMesh(self) and ishex(self.Hedra)



    def isPolyhedralMesh(self):
        """
        Returns whether the mesh is a polyhedral mesh or not

        Returns
        -------
        bool
            True if the mesh has polyhedral volumes, False otherwise
        """

        return self.isVolumetricMesh(self) and ispoly(self.Hedra)



    def genus(self):
        """
        Returns the genus of the mesh

        Returns
        -------
        int
            the genus of the mesh
        """

        return genus(self.Vertex,self.Edge,self.Face,self.Hedra)



    def update_ext(self):
        """Updates the external flags for vertices, edges, faces and volumes"""

        self.__compute_external_face()
        self.__compute_external_vertex()
        self.__compute_external_edge()
        self.__compute_external_hedra()



    def __compute_edge(self):
        """Computes the mesh edges"""

        if isempty(self.Face):
            return
        self.Edge = poly2edge(self.Face)[0]
        self.Edge = torch.t(unique(torch.t(torch.sort(self.Edge, dim=0)[0]),ByRows=True)[0])



    def __compute_external_vertex(self):
        """Computes the external vertices flags"""

        self.ExternalVertex = FalseTensor(row(self.Vertex),device=self.Vertex.device)
        if not self.isVolumetricMesh():
            self.ExternalVertex = TrueTensor(row(self.Vertex),device=self.Vertex.device)
            return
        J,I = poly2lin(self.Face)
        self.ExternalVertex = accumarray(J,self.ExternalFace[I])>=1
        self.ExternalVertex = self.ExternalVertex



    def __compute_external_edge(self):
        """Computes the external edges flags"""

        self.ExternalEdge = FalseTensor(col(self.Edge),device=self.Vertex.device)
        if not self.isVolumetricMesh():
            self.ExternalEdge = TrueTensor(col(self.Edge),device=self.Vertex.device)
            return
        self.ExternalEdge = self.ExternalVertex(self.Edge[0,:]) and self.ExternalVertex(self.Edge[1,:])



    def __compute_external_face(self):
        """Computes the external faces flags"""

        self.ExternalFace = FalseTensor(col(self.Face),device=self.Vertex.device)
        if not self.isVolumetricMesh():
            self.ExternalFace = TrueTensor(col(self.Face),device=self.Vertex.device)
            return
        J = poly2lin(self.Hedra)[0]
        self.ExternalFace = accumarray(J,1)==1
        self.ExternalFace = self.ExternalFace



    def __compute_external_hedra(self):
        """Computes the external volumes flags"""

        self.ExternalHedra = FalseTensor(col(self.Hedra),device=self.Vertex.device)
        if not self.isVolumetricMesh():
            return
        J,I = poly2ind(self.Hedra)
        self.ExternalHedra = accumarray(I,self.ExternalFace[J])>=1
        self.ExternalHedra = self.ExternalHedra
