from ..utility.find import *


class BaryCoord(torch.Tensor):
    """
    A class representing a barycentric coordinates.

    Methods
    -------
    is_valid()
        returns True if the coordinates sum up to 1, False otherwise
    is_invalid()
        returns True if the coordinates do not sum up to 1, False otherwise
    is_outside()
        returns True if the coordinates represent a point outside a polygon, False otherwise
    is_inside()
        returns True if the coordinates represent a point inside a polygon, False otherwise
    is_on_face()
        returns True if the coordinates represent a point inside a polygon, False otherwise
    is_on_edge()
        returns True if the coordinates represent a point inside the edge of a polygon, False otherwise
    is_on_vertex()
        returns True if the coordinates represent a point lying on the vertex of a polygon, False otherwise
    eval(P, T)
        returns the point represented by the coordinates from the given polygonal mesh
    """

    def __init__(self):
        super(BaryCoord, self).__init__()
        self.to(dtype=torch.float)

    def is_valid(self):
        """
        Returns True if the coordinates sum up to 1, False otherwise

        Returns
        -------
        Uint8Tensor
            True if the coordinates sum up to 1, False otherwise
        """

        return (torch.sum(self, -1, keepdim=True) == 1).squeeze()

    def is_invalid(self):
        """
        Returns True if the coordinates do not sum up to 1, False otherwise

        Returns
        -------
        Uint8Tensor
            True if the coordinates sum up to 1, False otherwise
        """

        return 1 - self.is_valid()

    def is_outside(self):
        """
        Returns True if the coordinates represent a point outside a polygon, False otherwise

        Returns
        -------
        Uint8Tensor
            True if the coordinates represent a point outside a polygon, False otherwise
        """

        return (torch.sum((self < 0) + (self > 1), -1, keepdim=True) > 0).squeeze()

    def is_inside(self):
        """
        Returns True if the coordinates represent a point inside a polygon, False otherwise

        Returns
        -------
        Uint8Tensor
            True if the coordinates represent a point inside a polygon, False otherwise
        """

        return 1 - self.is_outside()

    def is_on_face(self):
        """
        Returns True if the coordinates represent a point inside a polygon, False otherwise

        Returns
        -------
        Uint8Tensor
            True if the coordinates represent a point inside a polygon, False otherwise
        """
        return self.is_inside()

    def is_on_vertex(self):
        """
        Returns True if the coordinates represent a point lying on the vertex of a polygon, False otherwise

        Returns
        -------
        Uint8Tensor
            True if the coordinates represent a point lying on the vertex of a polygon, False otherwise
        """

        tf = self.is_on_face()
        bc = self == 1
        tf = tf * (torch.sum(bc, -1, keepdim=True) == 1).squeeze()
        if self.ndimension() == 1:
            bc = bc.unsqueeze(1)
            I = -torch.ones(1, dtype=torch.long, device=self.device)
        else:
            I = -torch.ones(self.shape[0], dtype=torch.long, device=self.device)
        i = find(bc[tf], linear=False).t().squeeze()
        I[i[0]] = i[1]
        return tf, I

    def is_on_edge(self):
        """
        Returns True if the coordinates represent a point inside the edge of a polygon, False otherwise

        Returns
        -------
        Uint8Tensor
            True if the coordinates represent a point inside the edge of a polygon, False otherwise
        """

        tf = self.is_on_face()
        bc = self == 0
        tf = tf * (torch.sum(bc, -1, keepdim=True) == 1).squeeze()
        if self.ndimension() == 1:
            bc = bc.unsqueeze(1)
            I  = -torch.ones(1, dtype=torch.long, device=self.device)
        else:
            I = -torch.ones(self.shape[0], dtype=torch.long, device=self.device)
        i = find(bc[tf], linear=False).t().squeeze()
        I[i[0]] = (i[1] + 1) % 3
        return tf, I

    def eval(self, P, T):
        """
        Returns the point represented by the coordinates from the given polygonal mesh

        Parameters
        ----------
        P : Tensor
            a (N,3,) point set tensor
        T : LongTensor
            a (V,M,) topology tensor

        Returns
        -------
        Tensor
            a (M,3,) point set tensor
        """

        return torch.sum(P[T] * (self if self.ndimension() > 1 else self.unsqueeze(0)).t().unsqueeze(P.ndimension()),
                         dim=0)
