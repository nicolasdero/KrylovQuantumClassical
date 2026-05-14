from .LanczosClassical import classical_Lanczos_algorithm_njit

import numpy as np
from numba import njit

@njit(inline = "always")
def L_c_coefficients(l: int, m: int) -> tuple[float, float, float, float, float, float]:
    """
    njited function that defines the coefficients that enter in the definition of the classical Liouvillian operator associated with the FP model.
    Note: the function uses the fact that the coefficients of that operator are separable in the l, m and k, n numbers of the vertex.

    Parameters
    ----------
    l: int
        The l number of the vertex.

    m: int
        The m number of the vertex.

    Returns
    -------
    (BP, BM, EP, EM, TP, TM): tuple[float, float, float, float, float, float]
        Tuple that contains the coefficients.
        
    Example
    -------
    >>> L_c_coefficients(1, 0)
    (1.4142135623730951, 1.4142135623730951, 1.0954451150103321, 1.0954451150103321, 0.0, 0.0)
    """
    lm_p = l + m
    lm_m = l - m

    denom_p = 1 / np.sqrt(2 * l + 3)
    denom_m = 1 / np.sqrt(2 * l - 1)

    BP = np.sqrt((lm_p + 1) * lm_m)
    BM = np.sqrt((lm_m + 1) * lm_p)

    EP = denom_p * np.sqrt((lm_p + 2) * (lm_p + 1))
    EM = denom_p * np.sqrt((lm_m + 2) * (lm_m + 1))

    TP = denom_m * np.sqrt((lm_p - 1) * lm_p)
    TM = denom_m * np.sqrt((lm_m - 1) * lm_m)

    return BP, BM, EP, EM, TP, TM


@njit(inline="always")
def vertex_index(l: int, m: int, k: int, n: int, k_block: int) -> int:
    """
    njited function that maps the vertex defined by the integers [l, m, k, n] (where -l <= m <= l and -k <= n <= k) to its corresponding index in the grid.

    Parameters
    ----------
    l: int
        The l number of the vertex.

    m: int
        The m number of the vertex.

    k: int
        The k number of the vertex.

    n: int
        The n number of the vertex.

    k_block: int
        The number of vertices contained in a fixed-l block of the grid, defined as (k_max + 1) ** 2.

    Returns
    -------
    index: int
        The index of the vertex in the grid.

    Example
    -------
    >>> vertex_index(1, 0, 1, 0, 4)
    10
    """
    return (l * l + l + m) * k_block + k * (k + 1) + n

@njit(fastmath=True)
def L_c(grid: np.ndarray, vertex: np.ndarray, c: float, index_current: int, k_max: int, a: float) -> None:
    """
    njited function that defines the action of the classical Liouvillian operator associated with the FP model on a given vertex of the grid. Its anayltical expression is given by Eq. (3.44) in the main text.
    Note: the function modifies the input grid in place.

    Parameters
    ----------
    grid: numpy.ndarray of shape (size_grid,)
        The grid on which the classical Liouvillian operator acts, where size_grid is the total number of vertices in the grid (see its definition in build_grid() below).

    vertex: numpy.ndarray of shape (4,)
        The vertex oon which the classical Liouvillian operator acts, defined by the quadruple of integers [l, m, k, n].

    c: float
        The coefficient associated to that vertex.

    index_current: int
        The index of that vertex in the grid.

    k_max: int
        The maximum k number of the vertices in the grid.

    a: float
        The parameter a that enters in the definition of the FP model.

    Example
    -------
    (see how it is used in the function classical_Lanczos_algorithm_njit() in LanczosClassical.py)
    """
    l, m, k, n = vertex

    k_block = (k_max + 1) ** 2

    grid[index_current] += -c * (1.0 + a) * (m + n)

    BP_lm, BM_lm, EP_lm, EM_lm, TP_lm, TM_lm = L_c_coefficients(l, m)
    BP_kn, BM_kn, EP_kn, EM_kn, TP_kn, TM_kn = L_c_coefficients(k, n)

    common = 1.0 - a
    a_k = common / np.sqrt(2 * k + 1)
    a_l = common / np.sqrt(2 * l + 1)
    c_ak = c * a_k
    c_al = c * a_l

    l1 = l
    m1 = m + 1
    if abs(m1) <= l1:

        k1 = k + 1
        n1 = n + 1
        if abs(n1) <= k1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += c_ak * BP_lm * EP_kn 

        k1 = k - 1
        n1 = n + 1
        if abs(n1) <= k1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += - c_ak * BP_lm * TM_kn

        k1 = k + 1
        n1 = n - 1
        if abs(n1) <= k1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += - c_ak * BP_lm * EM_kn

        k1 = k - 1
        n1 = n - 1
        if abs(n1) <= k1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += c_ak * BP_lm * TP_kn

    l1 = l
    m1 = m - 1
    if abs(m1) <= l1:

        k1 = k + 1
        n1 = n + 1
        if abs(n1) <= k1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += c_ak * BM_lm * EP_kn 

        k1 = k - 1
        n1 = n + 1
        if abs(n1) <= k1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += - c_ak * BM_lm * TM_kn 

        k1 = k + 1
        n1 = n - 1
        if abs(n1) <= k1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += - c_ak * BM_lm * EM_kn 

        k1 = k - 1
        n1 = n - 1
        if abs(n1) <= k1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += c_ak * BM_lm * TP_kn

    k1 = k
    n1 = n + 1
    if abs(n1) <= k1:

        l1 = l + 1
        m1 = m + 1
        if abs(m1) <= l1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += c_al * EP_lm * BP_kn

        l1 = l - 1
        m1 = m + 1
        if abs(m1) <= l1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += -c_al * TM_lm * BP_kn

        l1 = l + 1
        m1 = m - 1
        if abs(m1) <= l1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += -c_al * EM_lm * BP_kn

        l1 = l - 1
        m1 = m - 1
        if  abs(m1) <= l1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += c_al * TP_lm * BP_kn 

    k1 = k
    n1 = n - 1
    if abs(n1) <= k1:

        l1 = l + 1
        m1 = m + 1
        if abs(m1) <= l1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += c_al * EP_lm * BM_kn

        l1 = l - 1
        m1 = m + 1
        if abs(m1) <= l1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += -c_al * TM_lm * BM_kn

        l1 = l + 1
        m1 = m - 1
        if abs(m1) <= l1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += -c_al * EM_lm * BM_kn

        l1 = l - 1
        m1 = m - 1
        if  abs(m1) <= l1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += c_al * TP_lm * BM_kn

@njit
def build_grid(l_max: int, k_max: int) -> np.ndarray:
    """
    njited function that builds the grid of vertices for the classical FP model, defined as the set of all the vertices [l, m, k, n] such that 0 <= l <= l_max, -l <= m <= l, 0 <= k <= k_max and -k <= n <= k.

    Parameters
    ----------
    l_max: int
        The maximum l number of the vertices in the grid.

    k_max: int
        The maximum k number of the vertices in the grid.

    Returns
    -------
    grid: numpy.ndarray of shape (size_grid, 4)
        The grid of vertices, where size_grid is the total number of vertices in the grid, given by ((l_max + 1) ** 2) * ((k_max + 1) ** 2).

    Example
    -------
    >>> build_grid(1, 1)
    array([[ 0,  0,  0,  0], [ 0,  0,  1, -1], [ 0,  0,  1,  0], [ 0,  0,  1,  1], 
           [ 1, -1,  0,  0], [ 1, -1,  1, -1], [ 1, -1,  1,  0], [ 1, -1,  1,  1],
           [ 1,  0,  0,  0], [ 1,  0,  1, -1], [ 1,  0,  1,  0], [ 1,  0,  1,  1],
           [ 1,  1,  0,  0], [ 1,  1,  1, -1], [ 1,  1,  1,  0], [ 1,  1,  1,  1]])
    """
    size = ((l_max + 1) ** 2) * ((k_max + 1) ** 2)
    grid = np.empty((size, 4), dtype = np.int64)

    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            for k in range(k_max + 1):
                for n in range(-k, k + 1):
                    grid[idx, 0] = l
                    grid[idx, 1] = m
                    grid[idx, 2] = k
                    grid[idx, 3] = n
                    idx += 1

    return grid

def build_ic(ic: list, b_number: int) -> tuple[np.ndarray, int, int]:
    """
    Given initial conditions for the classical FP model, this function builds the initial grid. The function also builds the maximum l and k numbers of the vertices in that grid which depend on the initial conditions and the number of Lanczos iterations required.

    Parameters
    ----------
    ic: list 
        List containing the initial function decomposed on the spherical harmonics Y_l^m Z_k^n. The list has the structure [[[l_0, m_0, k_0, n_0], c_0], [l_1, m_1, k_1, n_1], c_1], ...] where c_i is the coefficient associated to the vertex defined by the quadruple of integers [l_i, m_i, k_i, n_i].

    b_number: int
        The number of Lanczos iterations that will be performed.

    Returns
    -------
    grid: numpy.ndarray of shape (size_grid_ic,)
        The initial grid, where size_grid_ic is the total number of vertices in the grid, given by ((l_max + 1) ** 2) * ((k_max + 1) ** 2), where l_max and k_max are the maximum l and k numbers of the vertices in the grid.

    l_max: int
        The maximum l number of the vertices in the grid.

    k_max: int
        The maximum k number of the vertices in the grid.

    Example
    -------
    >>> ic = [[[1, 0, 0, 0], 1.]]
    >>> build_ic(ic, 1)
    (array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
            1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
            0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
            0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
            0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]), np.int64(2), p.int64(1))
    """
    vertices_ic = np.array([v for v, _ in ic], dtype = np.int64)
    coeffs_ic  = np.array([c for _, c in ic], dtype = np.complex128)

    l_max = 0
    k_max = 0
    n_vertices_ic = vertices_ic.shape[0]

    for i in range(n_vertices_ic):
        l_max = max(l_max, vertices_ic[i, 0])
        k_max = max(k_max, vertices_ic[i, 2])

    l_max += b_number
    k_max += b_number

    k_block = (k_max + 1) ** 2
    size = ((l_max + 1) ** 2) * k_block
    grid = np.zeros(size, dtype = np.complex128)

    idx_ic = np.empty(n_vertices_ic, np.int64)
    val_ic = np.empty(n_vertices_ic, dtype=np.complex128)

    for i in range(n_vertices_ic):
        l, m, k, n = vertices_ic[i]
        j = vertex_index(l, m, k, n, k_block)
        grid[j] = coeffs_ic[i]
        idx_ic[i] = j
        val_ic[i] = coeffs_ic[i]

    return grid, l_max, k_max


class classicalFP():

    def __init__(self, a, ic, b_number):
        """
        Constructor for the classical FP class. Given the number of Lanczos coefficients to be computed and the initial conditions, this constructor builds the initial grid containing the initial function. It also initializes an empty grid explicitly containing the vertices [l, m, k, n], both of which are required to execute the classical Lanczos algorithm.

        Parameters
        ----------
        a: float
            The parameter a that enters in the definition of the FP model.

        ic: list
            List containing the initial function (see the precise structure of the list in the function build_ic() above).

        b_number: int
            The number of Lanczos iterations that will be performed.
        """
        self._a = a
        self._ic = ic
        self._b_number = b_number

        if self._b_number < 0:
            raise ValueError("The number of Lanczos iterations must be a non-negative integer.")

        print(type(self._ic))
        if not isinstance(self._ic, list):
            raise ValueError("The initial conditions must be a list.")
        elif self._ic == []:
            raise ValueError("The initial conditions cannot be an empty list.")
        else:
            for element in self._ic:
                if not isinstance(element, list) or len(element) != 2:
                    raise ValueError("Each element of the initial conditions list must be a list of the form [[l, m, k, n], c].")
                vertex, coeff = element
                if not isinstance(vertex, list) or len(vertex) != 4:
                    raise ValueError("The vertex in each element of the initial conditions list must be a list of four integers [l, m, k, n].")
                l, m, k, n = vertex
                if not all(isinstance(x, int) for x in vertex):
                    raise ValueError("The vertex in each element of the initial conditions list must be a list of four integers [l, m, k, n].")
                if abs(m) > l:
                    raise ValueError("The m number of each vertex in the initial conditions list must satisfy -l <= m <= l.")
                if abs(n) > k:
                    raise ValueError("The n number of each vertex in the initial conditions list must satisfy -k <= n <= k.")
                if l < 0 or k < 0:
                    raise ValueError("The l and k numbers of each vertex in the initial conditions list must be non-negative integers.")
                if not isinstance(coeff, (int, float, complex)):
                    raise ValueError("The coefficient in each element of the initial conditions list must be a number.")

        self._grid_ic, self._lmax, self._kmax = build_ic(self._ic, self._b_number)
        self._empty_grid = build_grid(self._lmax, self._kmax)

    @property
    def a(self):
        return self._a

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

    @property
    def kmax(self):
        return self._kmax

    def classical_Lanczos_algorithm(self):
        """
        Wrapper function that executes the classical Lanczos algorithm for the classical FP model by calling the njited function classical_Lanczos_algorithm_njit() defined in LanczosClassical.py.

        Example:
        --------
        >>> a = 0.
        >>> ic_zz = [[[1, 0, 1, 0], 1.]]
        >>> b_number = 20
        >>> FP_classical = classicalFP(a, ic_zz, b_number)
        >>> FP_classical.classical_Lanczos_algorithm()
        array([ 2.52982213,  3.47233968,  4.72013411,  5.66685709,  6.21286043,
                7.44283249,  7.4952765 ,  9.58306377,  9.4105079 , 11.11484156,
                11.3871763 , 12.48841138, 13.29722433, 14.09188443, 15.1276915 ,
                15.82689119, 16.80241057, 17.59512367, 18.44895755, 19.39593244])
        """
        return classical_Lanczos_algorithm_njit(self.grid_ic, self.empty_grid, L_c, self.b_number, self.kmax, self.a)
