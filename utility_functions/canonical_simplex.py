import math
import numpy as np


def simplex_projection_1d(vec):
    """
    Given a vector project it onto to the canonical simplex polyhedron in R^n

    Based on:
    Michelot, C., 1986. A finite algorithm for finding the projection of a point onto the canonical simplex of∝^n.
    Journal of Optimization Theory and Applications, 50(1), pp.195-200.

    Parameters
    ----------
    optimizer : numpy.ndarray
       A vector in R^N

    Returns
    -------
    numpy.ndarray
        The projection of the input vector onto the canonical simplex of R^n
    """
    if len(vec.shape) == 1:
        xt = vec.reshape((vec.shape[0], 1))
    else:
        xt = vec
    n = len(vec)
    In = np.ones((n, 1))
    for k in range(n):
        I = np.where(In == 0)[0]
        Ineg = np.where(In == 1)[0]
        dimIneg = len(Ineg)

        # project onto hyperplane
        x_til = np.zeros((n, 1))
        x_til[I] = 0
        x_til[Ineg] = xt[Ineg] - \
                         (np.matmul(np.ones((dimIneg, dimIneg)), xt[Ineg]) - np.ones((dimIneg, 1))) / dimIneg
        zero_idx = np.where(x_til < 0)
        if len(zero_idx) == 0:
            return x_til
        else:
            In[zero_idx] = 0
            xt = x_til
            xt[zero_idx] = 0
    return x_til


def simplex_projection(mat):
    """
    Calls a canonical simplex polyhedron projection for R^n for the the rowspace of the input Matrix mat

    Parameters
    ----------
    optimizer : numpy.ndarray
       A matrix in R^{M*N}

    Returns
    -------
    numpy.ndarray
        The projection of the rowspace onto the canonical simplex of R^n
    """
    if len(mat.shape) == 1:
        projection = simplex_projection_1d(mat)
        return projection.reshape((projection.shape[0],))

    n_rows = mat.shape[0]
    for i in range(n_rows):
        projection = simplex_projection_1d(mat[i])
        try:
            mat[i, :] = projection
        except:
            mat[i, :] = projection.reshape((projection.shape[0],))
    return mat
