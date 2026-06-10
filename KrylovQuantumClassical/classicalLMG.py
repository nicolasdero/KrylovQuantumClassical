from .LanczosClassical import extract_nonzero, classical_Lanczos_algorithm, classical_MC_Lanczos_algorithm

import numpy as np
from numba import njit, types
from numba.typed import Dict
import spherical
import quaternionic

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

@njit("void(complex128[:], int64[:], complex128, int64, float64, float64)", fastmath = True)
def L_c(grid: np.ndarray, vertex: np.ndarray, c: complex, index_current: int, h: float, J: float) -> None:
    """
    njited function that defines the action of the classical Liouvillian operator associated with the LMG model on a given vertex of the grid. Its anayltical expression is given by Eq. (3.41) in the main text.
    Note: the function modifies the input grid in place.

    Parameters
    ----------
    grid: numpy.ndarray of shape (size,)
        The grid on which the classical Liouvillian operator acts, where size is the total number of vertices in the grid (see its definition in build_grid() below).

    vertex: numpy.ndarray of shape (2,)
        The vertex on which the classical Liouvillian operator acts, defined by the quadruple of integers [l, m].

    c: float
        The coefficient associated to that vertex.

    index_current: int
        The index of that vertex in the grid.

    h: float
        The h parameter of the LMG model.

    J: float
        The J parameter of the LMG model.
        
    Example
    -------
    (see how it is used in the function LanczosClassical.classical_Lanczos_algorithm_njit())
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
    grid: numpy.ndarray of shape (size, 2)
        The grid of vertices, where size is the total number of vertices in the grid, given by ((l_max + 1) ** 2).

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
    Given initial conditions for the classical LMG model, this function builds the initial grid. The function also builds the maximum l number of the vertices in that grid which depend on the initial conditions and on the number of Lanczos iterations required.

    Parameters
    ----------
    ic: list 
        List containing the initial function decomposed on the spherical harmonics Y_l^m. The list has the structure [[[l_0, m_0], c_0], [l_1, m_1], c_1], ...] where c_i is the coefficient associated to the vertex defined by the pair of integers [l_i, m_i].

    b_number: int
        The number of Lanczos iterations that will be performed.

    Returns
    -------
    grid: numpy.ndarray of shape (size,)
       The initial grid, where size is the total number of vertices in the grid, given by ((l_max + 1) ** 2), where l_max is the maximum l number of the vertices in the grid.

    l_max: int
        The maximum l number of the vertices in the grid.

    Example
    -------
    >>> ic = [[[1, 0], 1.]]
    >>> build_ic(ic, 4)
    (array([0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
            0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
            0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
            0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
            0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]), np.int64(5))
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

@njit
def filter_points_LMG(h, J, E, delta_E, n_samples):

    cos_theta = np.random.uniform(-1.0, 1.0, size = n_samples)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, size = n_samples)

    H = - (J / 2) * np.cos(theta) ** 2 - h * np.sin(theta) * np.cos(phi)
    mask = np.abs(H - E) < delta_E / 2

    theta_filtered = theta[mask]
    phi_filtered = phi[mask]

    return theta_filtered, phi_filtered

def precompute_all_spherical_harmonics(l_max, h, J, E, delta_E, n_samples): 

    theta, phi = filter_points_LMG(h, J, E, delta_E, n_samples) 
    R = quaternionic.array.from_spherical_coordinates(theta, phi) 
    wigner = spherical.Wigner(l_max) 
    Y_all = wigner.sYlm(0, R) 
    n_theta = len(theta)

    return Y_all, n_theta

@njit(fastmath=True)
def create_IP_mat(l_max):

    G = np.full(((l_max + 1) ** 2, (l_max + 1) ** 2), - 1.0)

    for l1 in range(l_max + 1):
        for m1 in range(-l1, l1 + 1):
            for l2 in range(l_max + 1):
                for m2 in range(-l2, l2 + 1):
                    if (l1 + m1 + l2 + m2) % 2 == 1:
                        idx_1 = vertex_index(l1, m1)
                        idx_2 = vertex_index(l2, m2)
                        G[idx_1, idx_2] = 0.0

    return G

@njit
def grid_idx_map(empty_grid):
    size = empty_grid.shape[0]
    mapping = np.empty(size, dtype = np.int64)
    for i in range(size):
        l, m = empty_grid[i]
        mapping[i] = vertex_index(l, m)
    return mapping

@njit(fastmath=True)
def fast_dot_product(Y_C_tuple, Y_R_tuple, idx_i, idx_j, empty_grid):
    # 1. Model-specific parity selection rule check
    l_i, m_i = empty_grid[idx_i]
    l_j, m_j = empty_grid[idx_j]
    
    if (l_i + m_i + l_j + m_j) % 2 == 1:
        return 0.0

    # 2. Compute inner product if parity matches
    Y_C = Y_C_tuple[0]
    Y_R = Y_R_tuple[0]
    n_samples = Y_C.shape[0]
    accum = 0.0 + 0.0j
    
    for k in range(n_samples):
        accum += Y_C[k, idx_i] * Y_R[k, idx_j]
        
    return np.real(accum)

class classicalLMG():

    def __init__(self, h: float, J: float, ic: list):
        """
        Constructor for the classical LMG class.

        Parameters
        ----------
        h: float
            The h parameter of the LMG model.

        J: float
            The J parameter of the LMG model.

        ic: list
            List containing the initial function (see the precise structure of the list in the function build_ic() above).
        """
        self._h = h
        self._J = J
        self._ic = ic

        if not isinstance(self._ic, list):
            raise ValueError("The initial conditions must be a list.")
        elif self._ic == []:
            raise ValueError("The initial conditions cannot be an empty list.")
        else:
            for element in self._ic:
                if not isinstance(element, list) or len(element) != 2:
                    raise ValueError("Each element of the initial conditions list must be a list of the form [[l, m], c].")
                vertex, coeff = element
                if not isinstance(vertex, list) or len(vertex) != 2:
                    raise ValueError("The vertex in each element of the initial conditions list must be a list of two integers [l, m].")
                l, m = vertex
                if not all(isinstance(x, int) for x in vertex):
                    raise ValueError("The vertex in each element of the initial conditions list must be a list of two integers [l, m].")
                if l < 0:
                    raise ValueError("The l number of the vertex in the initial conditions list must be a non-negative integer.")
                if abs(m) > l:
                    raise ValueError("The m number of each vertex in the initial conditions list must satisfy -l <= m <= l.")
                if not isinstance(coeff, (int, float, complex)):
                    raise ValueError("The coefficient in each element of the initial conditions list must be a number.")

    @property
    def h(self):
        return self._h

    @property
    def J(self):
        return self._J

    @property
    def ic(self):
        return self._ic 

    def Lanczos_coeff_IT(self, b_number):
        """
        Wrapper function that executes the classical Lanczos algorithm for the classical LMG model by calling the njited function classical_Lanczos_algorithm_njit() defined in LanczosClassical.py.
        Given the number of Lanczos coefficients to be computed and the initial conditions, this constructor builds the initial grid containing the initial function. It also initializes an empty grid containing the vertices [l, m], both of which are required to execute the classical Lanczos algorithm.

        Parameters
        ----------
        b_number: int
            The number of Lanczos iterations that will be performed.

        Returns
        -------
        (see LanczosClassical.classical_Lanczos_algorithm_njit())

        Example:
        --------
        >>> h = 0.5
        >>> J = 1.0
        >>> ic_z = [[[1, 0], 1.]]
        >>> LMG_classical = classicalLMG(h, J, ic_z)
        >>> LMG_classical.Lanczos_coeff_IT(20)
        array([0.5       , 0.4472136 , 0.69178857, 1.42273149, 1.22647262,
               1.58192564, 2.12438249, 2.16357356, 2.40588715, 2.87951003,
               3.05058144, 3.28193812, 3.58260647, 4.06721226, 3.89773884,
               4.71668705, 4.51496265, 5.20782211, 5.05830963, 5.83685803])
        """
        if b_number < 0:
            raise ValueError("The number of Lanczos iterations must be a non-negative integer.")

        grid_ic, lmax = build_ic(self.ic, b_number)
        empty_grid = build_grid(lmax)
        return classical_Lanczos_algorithm(grid_ic, empty_grid, L_c, b_number, self.h, self.J)

    def __str__(self):
        terms = []

        for (l, m), coeff in self._ic:
            term = f"Y_{l}^{m}"
            terms.append((coeff, term))

        result = ""
        for i, (coeff, term) in enumerate(terms):
            if i == 0:
                if coeff < 0:
                    result += f"- {term}"
                else:
                    result += f"{term}"
            else:
                if coeff < 0:
                    result += f" - {term}"
                else:
                    result += f" + {term}"

        str_1 = f"Classical LMG model with parameters h = {self.h}, J = {self.J}." 
        str_2 = f"Initial conditions is {result}." 
        
        return str_1 + " " + str_2
    
class classicalLMG_MC():

    def __init__(self, h: float, J: float, ic: list, E: float, delta_E: float, n_samples: int):
        """
        Constructor for the classical and microcanonical LMG class.

        Parameters
        ----------
        h: float
            The h parameter of the LMG model.

        J: float
            The J parameter of the LMG model.

        ic: list
            List containing the initial function (see the precise structure of the list in the function build_ic() above).
        """
        self._h = h
        self._J = J
        self._ic = ic
        self._E = E
        self._delta_E = delta_E
        self._n_samples = n_samples

        if not isinstance(self._ic, list):
            raise ValueError("The initial conditions must be a list.")
        elif self._ic == []:
            raise ValueError("The initial conditions cannot be an empty list.")
        else:
            for element in self._ic:
                if not isinstance(element, list) or len(element) != 2:
                    raise ValueError("Each element of the initial conditions list must be a list of the form [[l, m], c].")
                vertex, coeff = element
                if not isinstance(vertex, list) or len(vertex) != 2:
                    raise ValueError("The vertex in each element of the initial conditions list must be a list of two integers [l, m].")
                l, m = vertex
                if not all(isinstance(x, int) for x in vertex):
                    raise ValueError("The vertex in each element of the initial conditions list must be a list of two integers [l, m].")
                if l < 0:
                    raise ValueError("The l number of the vertex in the initial conditions list must be a non-negative integer.")
                if abs(m) > l:
                    raise ValueError("The m number of each vertex in the initial conditions list must satisfy -l <= m <= l.")
                if not isinstance(coeff, (int, float, complex)):
                    raise ValueError("The coefficient in each element of the initial conditions list must be a number.")
                
        if self._delta_E <= 0:
            raise ValueError("The energy window delta_E must be a positive number.")

        if self._n_samples <= 0:
            raise ValueError("The number of samples must be a positive integer.")

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
    def E(self):
        return self._E

    @property
    def delta_E(self):
        return self._delta_E

    @property
    def n_samples(self):
        return self._n_samples

    def Lanczos_coeff_MC(self, b_number):
        if b_number < 0:
            raise ValueError("The number of Lanczos iterations must be a non-negative integer.")

        grid_ic, lmax = build_ic(self._ic, b_number)
        empty_grid = build_grid(lmax)

        grid_to_global_id = grid_idx_map(empty_grid)

        Y_all, n_theta = precompute_all_spherical_harmonics(lmax, self.h, self.J, self.E, self.delta_E, self.n_samples)
        
        Y_all_R = (np.asfortranarray(Y_all), )
        Y_all_C = (np.asfortranarray(np.conj(Y_all)), )
        
        G_mat = create_IP_mat(lmax)
        norm_MC = 1.0 / n_theta

        return classical_MC_Lanczos_algorithm(grid_ic, empty_grid, grid_to_global_id, L_c, G_mat, Y_all_C, Y_all_R, norm_MC, b_number, fast_dot_product, self.h, self.J)
    

#TO DO:

# Add a check for the energy window defined by E and delta_E in the constructor of classicalLMGMC. Do the same in the quantum algorithm.
# Add string representation for the classicalLMGMC class. It can be the same as the one for classicalLMG, but with an additional sentence that specifies the energy window defined by E and delta_E.