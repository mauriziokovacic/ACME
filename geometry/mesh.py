import torch
from ..utility.row         import *
from ..utility.col         import *
from ..utility.isempty     import *
from ..utility.unique      import *
from ..utility.accumarray  import *
from ..utility.FalseTensor import *
from ..utility.TrueTensor  import *
from ..topology.ispoly     import *
from ..topology.poly2edge  import *
from ..topology.poly2ind   import *
from ..topology.poly2lin   import *
from .genus                import *
from .mesh2data            import *


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
    num_vertices()
        Returns the number of vertices in the mesh
    num_edges()
        Returns the number of edges in the mesh
    num_faces()
        Returns the number of faces in the mesh
    num_hedras()
        Returns the number of volumes in the mesh
    is_empty()
        Returns wheter or not the mesh is empty
    recompute_normals()
        Recomputes the vertex normals
    is_surface_mesh()
        Returns True if the mesh has a surface
    is_tri_mesh()
        Returns True if the mesh is a triangle mesh
    is_quad_mesh()
        Returns True if the mesh is a quad mesh
    is_polygon_mesh()
        Returns True if the mesh is a generic polygonal mesh
    is_volumetric_mesh()
        Returns True if the mesh has volumes
    is_tet_mesh()
        Returns True if the mesh is a tetrahedral mesh
    is_hex_mesh()
        Returns True if the mesh is an hexaedral mesh
    is_polyhedral_mesh()
        Returns True if the mesh is a generic polyhedral mesh
    genus()
        Returns the genus of the mesh
    update_ext()
        Updates the external flags for vertices, edges, faces and volumes
    to_data()
        Returns a torch_geometric Data object
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

    def __init__(self, Vertex=None, Normal=None, UV=None, Edge=None, Face=None, Hedra=None, name=''):
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

    def num_vertices(self):
        """
        Returns the number of vertices in the mesh

        Returns
        -------
        int
            the number of vertices
        """

        return row(self.Vertex)

    def num_edges(self):
        """
        Returns the number of edges in the mesh

        Returns
        -------
        int
            the number of edges
        """

        return col(self.Edge)

    def num_faces(self):
        """
        Returns the number of faces in the mesh

        Returns
        -------
        int
            the number of faces
        """

        return col(self.Face)

    def num_hedras(self):
        """
        Returns the number of volumes in the mesh

        Returns
        -------
        int
            the number of volumes
        """

        return col(self.Hedra)

    def is_empty(self):
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

        self.Normal = vertex_normal(self.Vertex, self.Face)

    def is_surface_mesh(self):
        """
        Returns whether the mesh has a surface or not

        Returns
        -------
        bool
            True if the mesh has faces, False otherwise
        """

        return not isempty(self.Face)

    def is_tri_mesh(self):
        """
        Returns whether the mesh is a triangle mesh or not

        Returns
        -------
        bool
            True if the mesh has triangle faces, False otherwise
        """

        return self.is_surface_mesh() and istri(self.Face)

    def is_quad_mesh(self):
        """
        Returns whether the mesh is a quad mesh or not

        Returns
        -------
        bool
            True if the mesh has quad faces, False otherwise
        """

        return self.is_surface_mesh() and isquad(self.Face)

    def is_polygon_mesh(self):
        """
        Returns whether the mesh is a polygonal mesh or not

        Returns
        -------
        bool
            True if the mesh has polygonal faces, False otherwise
        """

        return self.is_surface_mesh() and ispoly(self.Face)

    def is_volumetric_mesh(self):
        """
        Returns whether the mesh has volumes or not

        Returns
        -------
        bool
            True if the mesh has volumes, False otherwise
        """

        return not isempty(self.Hedra)

    def is_tet_mesh(self):
        """
        Returns whether the mesh is a tetrahedral mesh or not

        Returns
        -------
        bool
            True if the mesh has tetrahedral volumes, False otherwise
        """

        return self.is_volumetric_mesh() and isquad(self.Hedra)

    def is_hex_mesh(self):
        """
        Returns whether the mesh is a haxaedral mesh or not

        Returns
        -------
        bool
            True if the mesh has hexaedral volumes, False otherwise
        """

        return self.is_volumetric_mesh() and ishex(self.Hedra)

    def is_polyhedral_mesh(self):
        """
        Returns whether the mesh is a polyhedral mesh or not

        Returns
        -------
        bool
            True if the mesh has polyhedral volumes, False otherwise
        """

        return self.is_volumetric_mesh() and ispoly(self.Hedra)

    def genus(self):
        """
        Returns the genus of the mesh

        Returns
        -------
        int
            the genus of the mesh
        """

        return genus(self.Vertex, self.Edge, self.Face, self.Hedra)

    def update_ext(self):
        """Updates the external flags for vertices, edges, faces and volumes"""

        self.__compute_external_face()
        self.__compute_external_vertex()
        self.__compute_external_edge()
        self.__compute_external_hedra()

    def to_data(self):
        """
        Returns a torch_geometric Data object

        Returns
        -------
        Data
            a torch_geometric Data object
        """

        return mesh2data(self.Vertex, self.Face, self.Normal, self.Edge)

    def __compute_edge(self):
        """Computes the mesh edges"""

        if isempty(self.Face):
            return
        self.Edge = poly2unique(poly2edge(self.Face)[0])[0]

    def __compute_external_vertex(self):
        """Computes the external vertices flags"""

        self.ExternalVertex = FalseTensor(row(self.Vertex), device=self.Vertex.device)
        if not self.is_volumetric_mesh():
            self.ExternalVertex = TrueTensor(row(self.Vertex), device=self.Vertex.device)
            return
        J, I = poly2lin(self.Face)
        self.ExternalVertex = (accumarray(J, self.ExternalFace[I]) >= 1).squeeze()

    def __compute_external_edge(self):
        """Computes the external edges flags"""

        self.ExternalEdge = FalseTensor(col(self.Edge), device=self.Vertex.device)
        if not self.is_volumetric_mesh():
            self.ExternalEdge = TrueTensor(col(self.Edge), device=self.Vertex.device)
            return
        self.ExternalEdge = self.ExternalVertex(self.Edge[0, :]) and self.ExternalVertex(self.Edge[1, :])
        self.ExternalEdge = self.ExternalEdge.squeeze()

    def __compute_external_face(self):
        """Computes the external faces flags"""

        self.ExternalFace = FalseTensor(col(self.Face), device=self.Vertex.device)
        if not self.is_volumetric_mesh():
            self.ExternalFace = TrueTensor(col(self.Face), device=self.Vertex.device)
            return
        J = poly2lin(self.Hedra)[0]
        self.ExternalFace = (accumarray(J, 1) == 1).squeeze()

    def __compute_external_hedra(self):
        """Computes the external volumes flags"""

        self.ExternalHedra = FalseTensor(col(self.Hedra), device=self.Vertex.device)
        if not self.is_volumetric_mesh():
            return
        J, I = poly2ind(self.Hedra)
        self.ExternalHedra = (accumarray(I, self.ExternalFace[J]) >= 1).suqeeze()
