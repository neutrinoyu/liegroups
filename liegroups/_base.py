from abc import ABCMeta, abstractmethod

class LieGroupBase(ABCMeta):
    """ Common abstract base class defining basic interface for Lie groups.
        Does not depend on any specific linear algebra library.
    """

    def __init__(self):
        pass

    @property
    @classmethod
    @abstractmethod
    def dim(cls):
        """Dimension of the underlying representation."""
        pass

    @property
    @classmethod
    @abstractmethod
    def dof(cls):
        """Underlying degrees of freedom (i.e., dimension of the tangent space)."""
        pass

    @abstractmethod
    def dot(self, other):
        """Multiply another group element or one or more vectors on the left.
        """
        pass

    @classmethod
    @abstractmethod
    def exp(cls, vec):
        """Exponential map for the group.

        Computes a transformation from a tangent vector.

        This is the inverse operation to log.
        """
        pass

    @classmethod
    @abstractmethod
    def identity(cls):
        """Return the identity transformation."""
        pass

    @abstractmethod
    def inv(self):
        """Return the inverse transformation."""
        pass

    @abstractmethod
    def log(self):
        """Logarithmic map for the group.

        Computes a tangent vector from a transformation.

        This is the inverse operation to exp.
        """
        pass

    @abstractmethod
    def normalize(self):
        """Normalize the group element to ensure it is valid and
        negate the effect of rounding errors.
        """
        pass

    @abstractmethod
    def perturb(self, vec):
        """Perturb the group element on the left by a vector in its local tangent space.
        """
        pass


class MatrixLieGroupBase(LieGroupBase):
    """Common abstract base class defining basic interface for Matrix Lie Groups.
       Does not depend on any specific linear algebra library.
    """

    def __repr__(self):
        """Return a string representation of the transformation."""
        return "<{}.{}>\n{}".format(self.__class__.__module__, self.__class__.__name__, self.as_matrix()).replace("\n", "\n| ")

    @abstractmethod
    def adjoint(self):
        """Return the adjoint matrix of the transformation."""
        pass

    @abstractmethod
    def as_matrix(self):
        """Return the matrix representation of the transformation."""
        pass

    @classmethod
    @abstractmethod
    def from_matrix(cls, mat, normalize=False):
        """Create a transformation from a matrix (safe, but slower)."""
        pass

    @classmethod
    @abstractmethod
    def inv_left_jacobian(cls, vec):
        """Inverse of the left Jacobian for the group."""
        pass

    @classmethod
    @abstractmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid transformation matrix."""
        pass

    @classmethod
    @abstractmethod
    def left_jacobian(cls, vec):
        """Left Jacobian for the group."""
        pass

    @classmethod
    @abstractmethod
    def vee(cls, mat):
        """vee operator as defined by Barfoot.

        This is the inverse operation to wedge.
        """
        pass

    @classmethod
    @abstractmethod
    def wedge(cls, vec):
        """wedge operator as defined by Barfoot.

        This is the inverse operation to vee.
        """
        pass


class SOMatrixBase(MatrixLieGroupBase):
    """Common abstract base class for Special Orthogonal Matrix Lie Groups SO(N).
       Does not depend on any specific linear algebra library.
    """

    def __init__(self, mat):
        """Create a transformation from a rotation matrix (unsafe, but faster)."""
        self.mat = mat
        """Storage for the rotation matrix."""

    def as_matrix(self):
        """Return the matrix representation of the rotation."""
        return self.mat

    def perturb(self, phi):
        """Perturb the rotation in-place on the left by a vector in its local tangent space.

        .. math::
            \\mathbf{C} \\gets \\exp(\\boldsymbol{\\phi}^\\wedge) \\mathbf{C}
        """
        self.mat = self.__class__.exp(phi).dot(self).mat


class SEMatrixBase(MatrixLieGroupBase):
    """Common abstract base class for Special Euclidean Matrix Lie Groups SE(N).
       Does not depend on any specific linear algebra library.
    """

    def __init__(self, rot, trans):
        """Create a transformation from a translation and a rotation (unsafe, but faster)"""
        self.rot = rot
        """Storage for the rotation matrix."""
        self.trans = trans
        """Storage for the translation vector."""

    @classmethod
    @abstractmethod
    def odot(cls, p, directional=False):
        """odot operator as defined by Barfoot."""
        pass

    def perturb(self, xi):
        """Perturb the transformation in-place on the left by a vector in its local tangent space.

        .. math::
            \\mathbf{T} \\gets \\exp(\\boldsymbol{\\xi}^\\wedge) \\mathbf{T}
        """
        perturbed = self.__class__.exp(xi).dot(self)
        self.rot = perturbed.rot
        self.trans = perturbed.trans

    @property
    @classmethod
    @abstractmethod
    def RotationType(cls):
        """Rotation type."""
        pass


class VectorLieGroupBase(LieGroupBase):
    """Common abstract base class for Lie Groups with vector parametrizations 
       (complex, quaternions, dual quaternions). Does not depend on any  
       specific linear algebra library.
    """

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        """Return a string representation of the transformation."""
        return "<{}.{}>\n{}".format(self.__class__.__module__, self.__class__.__name__, self.data).replace("\n", "\n| ")

    @abstractmethod
    def conjugate(self):
        """Return the conjugate of the vector"""
        pass


class SOMatrixBase(_base.SOMatrixBase):
    """Implementation of methods common to SO(N) matrix lie groups using Numpy"""

    def dot(self, other):
        """Multiply another rotation or one or more vectors on the left.
        """
        if isinstance(other, self.__class__):
            # Compound with another rotation
            return self.__class__(np.dot(self.mat, other.mat))
        else:
            other = np.atleast_2d(other)

            # Transform one or more 2-vectors or fail
            if other.shape[1] == self.dim:
                return np.squeeze(np.dot(self.mat, other.T).T)
            else:
                raise ValueError(
                    "Vector must have shape ({},) or (N,{})".format(self.dim, self.dim))

    @classmethod
    def identity(cls):
        """Return the identity rotation."""
        return cls(np.identity(cls.dim))

    def inv(self):
        """Return the inverse rotation:

        .. math::
            \\mathbf{C}^{-1} = \\mathbf{C}^T
        """
        return self.__class__(self.mat.T)

    @classmethod
    def from_matrix(cls, mat, normalize=False):
        """Create a rotation from a matrix (safe, but slower).

        Throws an error if mat is invalid and normalize=False.
        If normalize=True invalid matrices will be normalized to be valid.
        """
        mat_is_valid = cls.is_valid_matrix(mat)

        if mat_is_valid or normalize:
            result = cls(mat)
            if not mat_is_valid and normalize:
                result.normalize()
        else:
            raise ValueError(
                "Invalid rotation matrix. Use normalize=True to handle rounding errors.")

        return result

    @classmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid rotation matrix."""
        return mat.shape == (cls.dim, cls.dim) and \
            np.isclose(np.linalg.det(mat), 1.) and \
            np.allclose(mat.T.dot(mat), np.identity(cls.dim))

    def normalize(self):
        """Normalize the rotation matrix to ensure it is valid and
        negate the effect of rounding errors.
        """
        # The SVD is commonly written as a = U S V.H.
        # The v returned by this function is V.H and u = U.
        U, _, V = np.linalg.svd(self.mat, full_matrices=False)

        S = np.identity(self.dim)
        S[self.dim - 1, self.dim - 1] = np.linalg.det(U) * np.linalg.det(V)

        self.mat = U.dot(S).dot(V)


class SEMatrixBase(_base.SEMatrixBase):
    """Implementation of methods common to SE(N) matrix lie groups using Numpy"""

    def as_matrix(self):
        """Return the matrix representation of the rotation."""
        R = self.rot.as_matrix()
        t = np.reshape(self.trans, (self.dim - 1, 1))
        bottom_row = np.append(np.zeros(self.dim - 1), 1.)
        return np.vstack([np.hstack([R, t]),
                          bottom_row])

    def dot(self, other):
        """Multiply another rotation or one or more vectors on the left.
        """
        if isinstance(other, self.__class__):
            # Compound with another transformation
            return self.__class__(self.rot.dot(other.rot),
                                  self.rot.dot(other.trans) + self.trans)
        else:
            other = np.atleast_2d(other)

            if other.shape[1] == self.dim - 1:
                # Transform one or more 2-vectors
                return np.squeeze(self.rot.dot(other) + self.trans)
            elif other.shape[1] == self.dim:
                # Transform one or more 3-vectors
                return np.squeeze(self.as_matrix().dot(other.T)).T
            else:
                raise ValueError("Vector must have shape ({},), ({},), (N,{}) or (N,{})".format(
                    self.dim - 1, self.dim, self.dim - 1, self.dim))

    @classmethod
    def from_matrix(cls, mat, normalize=False):
        """Create a transformation from a matrix (safe, but slower).

        Throws an error if mat is invalid and normalize=False.
        If normalize=True invalid matrices will be normalized to be valid.
        """
        mat_is_valid = cls.is_valid_matrix(mat)

        if mat_is_valid or normalize:
            result = cls(
                cls.RotationType(mat[0:cls.dim - 1, 0:cls.dim - 1]),
                mat[0:cls.dim - 1, cls.dim - 1])
            if not mat_is_valid and normalize:
                result.normalize()
        else:
            raise ValueError(
                "Invalid transformation matrix. Use normalize=True to handle rounding errors.")

        return result

    @classmethod
    def identity(cls):
        """Return the identity transformation."""
        return cls.from_matrix(np.identity(cls.dim))

    def inv(self):
        """Return the inverse transformation:

        .. math::
            \\mathbf{T}^{-1} = 
                \\begin{bmatrix}
                    \\mathbf{C}^T & -\\mathbf{C}^T\\mathbf{r} \\\\
                    \\mathbf{0}^T & 1
                \\end{bmatrix}
        """
        inv_rot = self.rot.inv()
        inv_trans = -(inv_rot.dot(self.trans))
        return self.__class__(inv_rot, inv_trans)

    @classmethod
    def is_valid_matrix(cls, mat):
        """Check if a matrix is a valid transformation matrix."""
        bottom_row = np.append(np.zeros(cls.dim - 1), 1.)

        return mat.shape == (cls.dim, cls.dim) and \
            np.array_equal(mat[cls.dim - 1, :], bottom_row) and \
            cls.RotationType.is_valid_matrix(mat[0:cls.dim - 1, 0:cls.dim - 1])

    def normalize(self):
        """Normalize the transformation matrix to ensure it is valid and
        negate the effect of rounding errors.
        """
        self.rot.normalize()


class VectorLieGroupBase(_base.VectorLieGroupBase):
    """Implementation of methods common to vector-parametrized lie groups using Numpy"""

    def normalize(self):
        self.data = self.data / np.linalg.norm(self.data)

    def conjugate(self):
        return self.__class__(np.hstack([self.data[0], -self.data[1:]]))
