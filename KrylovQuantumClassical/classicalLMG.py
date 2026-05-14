from .LanczosClassical import classical_Lanczos_algorithm_njit

import numpy as np
from numba import njit

@njit(inline = "always")
def L_c_coefficients(l: int, m: int) -> tuple[float, float, float, float]:
    """
    njited function that defines the coefficients that enter in the definition of the classical Liouvillian operator associated with the LMG model.

    Parameters
    ----------
    l: int
        The l number of the vertex.

    m: int
        The m number of the vertex.

    Returns
    -------
    (C1, C2, C3, C4): tuple[float, float, float, float]
        Tuple that contains the coefficients.
        
    Example
    -------
    >>> L_c_coefficients(1, 1)
    (0.44721359549995804, 0.0, 0.0, 0.7071067811865476)
    """
    lm_p = l + m
    lm_m = l - m
    m_scaled = m / np.sqrt(2 * l + 1)

    C1 = m_scaled * np.sqrt(((lm_p + 1) * (lm_m + 1)) / (2 * l + 3))
    C2 = m_scaled * np.sqrt((lm_m * lm_p) / (2 * l - 1))
    C3 = 0.5 * np.sqrt((lm_p + 1) * lm_m)
    C4 = 0.5 * np.sqrt((lm_m + 1) * lm_p)

    return C1, C2, C3, C4

@njit(inline="always")
def vertex_index(l: int, m: int) -> int:
    """
    njited function that maps the vertex defined by the pair of integers [l, m] (where -l <= m <= l) to its corresponding index in the grid.

    Parameters
    ----------
    l: int
        The l number of the vertex.

    m: int
        The m number of the vertex.

    Returns
    -------
    index: int
        The index of the vertex in the grid

    Example
    -------
    >>> vertex_index(1, 0)
    2
    """
    return l * (l + 1) + m

#@njit("void(float64[:], int64[:], float64, int64, float64, int64)", fastmath = True)
@njit(fastmath = True)
def L_c(grid, vertex, c, index_current, h, J):
    """
    njted function that applies the classical Liouvillian operator associated with the LMG model to a given vertex of the grid. As it acts on a given vertex, the function modifies the grid.

    Parameters
    ----------
    grid: bla bla
        bla bla

    vertex: numpy.ndarray of shape (2,)
        The vertex is defined by the pair of integers [l, m].

    c: float
        The coefficient associated to that vertex.

    index_current: int
        The index of the current vertex in the grid.

    param: list of length 3
        List that contains parameters: param[0] is the h parameter, param[1] is the J parameter of the LMG model and param[2] is the size of the grid.

    Returns
    -------
    grid: numpy.ndarray of shape (size_grid,)
        Grid obtained after the application of the classical Liouvillian operator to the vertex.
        
    Example
    -------
    >>> x = L_c_LMG([1, 0], 1, 2, np.array([1, 1, 36]))
    >>> print(x[np.where(x != 0)])
    [0.70710678, 0.70710678]
    """
    l, m = vertex
    C1, C2, C3, C4 = L_c_coefficients(l, m)

    c_J = - c * J
    c_h = - c * h

    grid[index_current] = 0

    l1 = l + 1
    m1 = m
    if abs(m1) <= l1:
        idx = vertex_index(l1, m1)
        grid[idx] += c_J * C1

    l1 = l - 1
    m1 = m
    if abs(m1) <= l1:
        idx = vertex_index(l1, m1)
        grid[idx] += c_J * C2

    l1 = l
    m1 = m + 1
    if abs(m1) <= l1:
        idx = vertex_index(l1, m1)
        grid[idx] += c_h * C3

    l1 = l
    m1 = m - 1
    if abs(m1) <= l1:
        idx = vertex_index(l1, m1)
        grid[idx] += c_h * C4

@njit
def build_grid(l_max: int) -> np.ndarray:
    """
    njited function that builds the grid of vertices for the classical LMG model, defined as the set of all the vertices [l, m] such that 0 <= l <= l_max and -l <= m <= l.

    Parameters
    ----------
    l_max: int
        The maximum l number of the vertices in the grid.

    Returns
    -------
    grid: numpy.ndarray of shape (size_grid, 2)
        The grid of vertices, where size_grid is the total number of vertices in the grid, given by ((l_max + 1) ** 2).

    Example
    -------
    >>> build_grid(2)
    array([[ 0,  0], [ 1, -1], [ 1,  0], [ 1,  1], [ 2, -2], [ 2, -1], [ 2,  0], [ 2,  1], [ 2,  2]])
    """
    size = ((l_max + 1) ** 2)
    grid = np.empty((size, 2), dtype = np.int64)

    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            grid[idx, 0] = l
            grid[idx, 1] = m
            idx += 1

    return grid

def build_ic(ic, b_number):
    """
    ic: list whose structure is of the form [[[l_0, m_0], c_0], [l_1, m_1], c_1], ...].
        The initial function decomposed in terms of the spherical harmonics Y_l^m.
    """
    vertices_ic = np.array([v for v, _ in ic], dtype = np.int64)
    coeffs_ic  = np.array([c for _, c in ic], dtype = np.complex128)

    l_max = 0
    n_vertices_ic = vertices_ic.shape[0]

    for i in range(n_vertices_ic):
        l_max = max(l_max, vertices_ic[i, 0])

    l_max += b_number

    size = ((l_max + 1) ** 2)
    grid = np.zeros(size, dtype = np.complex128)

    idx_ic = np.empty(n_vertices_ic, np.int64)
    val_ic = np.empty(n_vertices_ic, dtype=np.complex128)

    for i in range(n_vertices_ic):
        l, m = vertices_ic[i]
        j = vertex_index(l, m)
        grid[j] = coeffs_ic[i]
        idx_ic[i] = j
        val_ic[i] = coeffs_ic[i]

    return grid, l_max

class classicalLMG():

    def __init__(self, h, J, ic, b_number):
        super().__init__()
        self._h = h
        self._J = J
        self._ic = ic
        self._b_number = b_number
        self._grid_ic, self._lmax = build_ic(self._ic, self._b_number)
        self._empty_grid = build_grid(self._lmax)

    @property
    def h(self):
        return self._h

    @property
    def J(self):
        return self._J

    @property
    def ic(self):
        return self._ic 

    @property
    def b_number(self):
        return self._b_number

    @property
    def grid_ic(self):
        return self._grid_ic

    @property
    def empty_grid(self):
        return self._empty_grid

    @property
    def lmax(self):
        return self._lmax


    def classical_Lanczos_algorithm(self):

        return classical_Lanczos_algorithm_njit(self.grid_ic, self.empty_grid, L_c, self.b_number, self.h, self.J)