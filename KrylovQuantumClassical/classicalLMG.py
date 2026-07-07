from .LanczosClassical import extract_nonzero, classical_Lanczos_algorithm, classical_MC_Lanczos_algorithm

import numpy as np
from numba import njit
import scipy.special as spc
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
    njited function that defines the action of the classical Liouvillian operator associated with the LMG model on a given vertex of the grid. Its analytical expression is given by Eq. (3.39) in the main text.
    Note: the function modifies the input grid in place.

    Parameters
    ----------
    grid: numpy.ndarray of shape (size,)
        The grid on which the classical Liouvillian operator acts, where size is the total number of vertices in the grid (see its definition in build_grid() below).

    vertex: numpy.ndarray of shape (2,)
        The vertex on which the classical Liouvillian operator acts, defined by the pair of integers [l, m].

    c: complex
        The coefficient associated with that vertex.

    index_current: int
        The index of that vertex in the grid.

    h: float
        The h parameter of the LMG model.

    J: float
        The J parameter of the LMG model.
        
    Example
    -------
    (see how it is used in the function LanczosClassical.classical_Lanczos_algorithm() or in LanczosClassical.classical_MC_Lanczos_algorithm())
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

def build_ic(ic: list, b_number: int) -> tuple[np.ndarray, int]:
    """
    Given initial conditions for the classical LMG model, this function builds the initial grid. The function also builds the maximum l number of the vertices in that grid which depends on the initial conditions and on the number of Lanczos iterations required.

    Parameters
    ----------
    ic: list 
        List containing the initial function decomposed on the spherical harmonics Y_l^m. The list has the structure [[[l_0, m_0], c_0], [[l_1, m_1], c_1], ...] where c_i is the coefficient associated to the vertex defined by the pair of integers [l_i, m_i].

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

    for i in range(n_vertices_ic):
        l, m = vertices_ic[i]
        j = vertex_index(l, m)
        grid[j] = coeffs_ic[i]

    return grid, l_max

@njit
def filter_points_LMG(h: float, J: float, E: float, delta_E: float, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    njited function that generates random points on the sphere and filters them based on the selected energy shell defined by E and delta_E. 
    The energy is given by the classical LMG Hamiltonian, see Eq. (3.38) in the main text.

    Parameters
    ----------
    h: float
        The h parameter of the LMG model.  

    J: float
        The J parameter of the LMG model.

    E: float
        The energy value that defines the center of the energy shell.

    delta_E: float
        The width of the energy shell (Note: the energy shell is defined as the interval [E - delta_E / 2, E + delta_E / 2]).

    n_samples: int
        The number of random points to be generated on the sphere.

    Returns
    -------
    theta_filtered: numpy.ndarray
        The filtered polar angles of the points lying within the energy shell.

    phi_filtered: numpy.ndarray
        The filtered azimuthal angles of the points lying within the energy shell.

    Example
    -------
    >>> h = 0.5
    >>> J = 1.
    >>> E = 0.
    >>> delta_E = 0.1
    >>> n_samples = 10 ** 2
    >>> filter_points_LMG(h, J, E, delta_E, n_samples)
    (array([2.15590292, 0.85595824, 0.65767711, 1.00849223, 1.63762661, 2.24664296, 2.06041061, 0.7672643 ]),
     array([4.22440124, 2.24897562, 3.17796324, 4.38462869, 4.76000994, 2.05027066, 4.38360706, 3.71587467]))
    """
    cos_theta = np.random.uniform(-1.0, 1.0, size = n_samples)
    theta = np.arccos(cos_theta)
    phi = np.random.uniform(0, 2 * np.pi, size = n_samples)

    H = - (J / 2) * np.cos(theta) ** 2 - h * np.sin(theta) * np.cos(phi)
    mask = np.abs(H - E) < delta_E / 2

    theta_filtered = theta[mask]
    phi_filtered = phi[mask]

    return theta_filtered, phi_filtered

def precompute_all_spherical_harmonics_1(l_max: int, h: float, J: float, E: float, delta_E: float, n_samples: int) -> tuple[np.ndarray, int]: 
    """
    Given the maximum l number of the spherical harmonics, the parameters of the classical LMG model, and the energy shell defined by E and delta_E, this function precomputes all the spherical harmonics for all the points that lie within the energy shell (sampled using the function filter_points_LMG()).
    Note that the ordering of the spherical harmonics is given by the ordering of the vertices in the grid, see build_grid() above.

    Parameters
    ---------- 
    l_max: int
        The maximum l number of the spherical harmonics.
    
    h: float
        The h parameter of the LMG model.  

    J: float
        The J parameter of the LMG model.

    E: float
        The energy value that defines the center of the energy shell.

    delta_E: float
        The width of the energy shell (Note: the energy shell is defined as the interval [E - delta_E / 2, E + delta_E / 2]).

    n_samples: int
        The number of random points to be generated on the sphere.

    Returns
    -------
    Y_all: numpy.ndarray of shape (n_theta, (l_max + 1) ** 2)
        The array containing all the spherical harmonics evaluated at the points lying within the energy shell, where n_theta is the number of points contained in the shell.

    n_theta: int
        The number of points contained in the energy shell.

    Example
    -------
    >>> l_max = 1
    >>> h = 0.5
    >>> J = 1.
    >>> E = 0.
    >>> delta_E = 0.1
    >>> n_samples = 10 ** 2
    >>> precompute_all_spherical_harmonics_1(l_max, h, J, E, delta_E, n_samples)
    (array([[ 0.28209479+0.j, -0.20779877-0.02366884j, -0.38891043+0.j,   0.20779877-0.02366884j],
            [ 0.28209479+0.j, -0.05119343-0.31658299j, -0.18177265+0.j,   0.05119343-0.31658299j],
            [ 0.28209479+0.j, -0.18297038-0.10451829j, -0.38720531+0.j,   0.18297038-0.10451829j],
            [ 0.28209479+0.j, -0.09970728+0.26303115j,  0.28368744+0.j,   0.09970728+0.26303115j],
            [ 0.28209479+0.j,  0.00867739+0.34466379j,  0.03155248+0.j,  -0.00867739+0.34466379j],
            [ 0.28209479+0.j, -0.08664975+0.27912522j,  0.26056531+0.j,   0.08664975+0.27912522j],
            [ 0.28209479+0.j,  0.0037707 -0.33962176j, -0.08954387+0.j,  -0.0037707 -0.33962176j],
            [ 0.28209479+0.j, -0.20859189+0.07526107j,  0.37467694+0.j,   0.20859189+0.07526107j],
            [ 0.28209479+0.j, -0.05354956+0.2996404j , -0.2311462 +0.j,   0.05354956+0.2996404j]]), 9)
    """
    theta, phi = filter_points_LMG(h, J, E, delta_E, n_samples) 
    R = quaternionic.array.from_spherical_coordinates(theta, phi) 
    wigner = spherical.Wigner(l_max) 
    Y_all = wigner.sYlm(0, R) 
    n_theta = len(theta)

    return Y_all, n_theta

@njit(fastmath=True)
def create_IP_mat(l_max: int) -> np.ndarray:
    """
    Function that creates the microcanonical inner product matrix for the classical LMG model. The shape of the matrix corresponds to the number of vertices in the grid.
    Note that at this stage, the matrix is initialized with −1, except for entries forbidden by the parity selection rule, which are initialized to 0. It is later filled with the microcanonical inner products (Y_l^m, Y_l'^m')_{E,delta_E}, at position (vertex_index(l, m), vertex_index(l', m')) in the matrix. 
    The inner product is zero if the parity selection rule is not satisfied, i.e., if l + m + l' + m' is odd, see Eq. (3.45b) in the main text.

    Parameters
    ----------
    l_max: int
        The maximum l number of the vertices in the grid.

    Returns
    -------
    G: numpy.ndarray of shape ((l_max + 1) ** 2, (l_max + 1) ** 2)
        The microcanonical inner product matrix for the classical LMG model.

    Example
    -------
    >>> l_max = 1
    >>> create_IP_mat(l_max)
    array([[-1., -1.,  0., -1.],
           [-1., -1.,  0., -1.],
           [ 0.,  0., -1.,  0.],
           [-1., -1.,  0., -1.]])
    """
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

@njit(fastmath=True)
def dot_product(Y_C_tuple: tuple[np.ndarray], Y_R_tuple: tuple[np.ndarray], vertex_i: np.ndarray, vertex_j: np.ndarray) -> float:
    """
    njited function that computes the unnormalized microcanonical inner product (Y_l^m, Y_l'^m')_{E,delta_E} between two spherical harmonics Y_l^m and Y_l'^m' using Monte Carlo sampling. The inner product is computed as a sum over the points in the energy shell.
    The normalization factor is included later in LanczosClassical.IP_MC().

    Parameters
    ----------  
    Y_C_tuple: tuple of numpy.ndarray of shape ((n_theta, (l_max + 1) ** 2), )
        The tuple containing the array of all the conjugated spherical harmonics evaluated at the points lying within the energy shell, where n_theta is the number of points contained in the shell.
        Note: the tuple format is required for the function classical_MC_Lanczos_algorithm(), defined in LanczosClassical.py, to be compatible with the FP model, which has two arrays of spherical harmonics (and for further generalizations of the classical models that may require more than one array of spherical harmonics).

    Y_R_tuple: tuple of numpy.ndarray of shape ((n_theta, (l_max + 1) ** 2), )
        The tuple containing the array of all the spherical harmonics evaluated at the points lying within the energy shell, where n_theta is the number of points contained in the shell.
        Note: the tuple format is required for the function classical_MC_Lanczos_algorithm(), defined in LanczosClassical.py, to be compatible with the FP model, which has two arrays of spherical harmonics (and for further generalizations of the classical models that may require more than one array of spherical harmonics).

    vertex_i: numpy.ndarray of shape (2,)
        Active vertex in the grid corresponding to [l, m]. An active vertex corresponds to a vertex over which the classical Liouvillian operator acts.

    vertex_j: numpy.ndarray of shape (2,)
        Active vertex in the grid corresponding to [l', m']. An active vertex corresponds to a vertex over which the classical Liouvillian operator acts.

    Returns
    -------
    result_dot_product: float
        The value of the unnormalized microcanonical inner product (Y_l^m, Y_l'^m')_{E,delta_E} between the two spherical harmonics Y_l^m and Y_l'^m' computed using Monte Carlo sampling.

    Example
    -------
    >>> l_max = 1
    >>> h = 0.5
    >>> J = 1.
    >>> E = 0.
    >>> delta_E = 0.1
    >>> n_samples = 10 ** 2
    >>> Y_all, n_theta = KQC.classicalLMG.precompute_all_spherical_harmonics_1(l_max, h, J, E, delta_E, n_samples)
    >>> vertex_i = [1, 0]
    >>> vertex_j = [1, 0]
    >>> KQC.classicalLMG.dot_product((Y_all,), (Y_all,), vertex_i, vertex_j)
    0.5434625148197845
    """
    l_i, m_i = vertex_i
    l_j, m_j = vertex_j
    
    if (l_i + m_i + l_j + m_j) % 2 == 1:
        return 0.0

    Y_C = Y_C_tuple[0]
    Y_R = Y_R_tuple[0]
    n_samples = Y_C.shape[0]
    accum = 0.0 + 0.0j
    
    for k in range(n_samples):
        accum += Y_C[k, l_i * (l_i + 1) + m_i] * Y_R[k, l_j * (l_j + 1) + m_j]

    result_dot_product = np.real(accum)
        
    return result_dot_product

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
    
    def spectral_width(self) -> float:
        """
        Computes the spectral width of the classical LMG model, defined as the difference between the maximum and minimum energies of the classical LMG Hamiltonian, see Eq. (B.19) in the main text.

        Returns
        -------
        spectral_width: float
            The spectral width of the classical LMG model.

        Example
        -------
        >>> h = 0.5
        >>> J = 1.
        >>> ic_z = [[[1, 0], 1.]]
        >>> LMG_classical = classicalLMG(h, J, [[[1, 0], 1.]])
        >>> LMG_classical.spectral_width()
        1.125
        """
        if self._h <= self._J:
            spectral_width =  ((self._h + self._J) ** 2) / (2 * self._J)
        else:
            spectral_width = 2 * self._h

        return spectral_width

    def Lanczos_coeff_IT(self, b_number: int) -> np.ndarray:
        """
        Wrapper function that executes the classical Lanczos algorithm for the classical LMG model by calling the njited function classical_Lanczos_algorithm() defined in LanczosClassical.py.
        Given the number of Lanczos coefficients to be computed and the initial conditions, this method builds the initial grid containing the initial function. It also initializes an empty grid containing the vertices [l, m], both of which are required to execute the classical Lanczos algorithm.

        Parameters
        ----------
        b_number: int
            The number of Lanczos iterations that will be performed.

        Returns
        -------
        (see LanczosClassical.classical_Lanczos_algorithm())

        Example
        -------
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
        str_2 = f"Initial condition is {result}." 
        
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

        E: float
            The energy value that defines the center of the energy shell.

        delta_E: float
            The width of the energy shell (Note: the energy shell is defined as the interval [E - delta_E / 2, E + delta_E / 2]).

        n_samples: int
            The number of random points to be generated on the sphere.
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
        
        if self._h <= self._J:
            if self._E < - (self._h ** 2 + self._J ** 2) / (2 * self._J) or self._E > self._h:
                raise ValueError(f"For h <= J, the energy E must be in the interval [{-(self._h ** 2 + self._J ** 2) / (2 * self._J)}, {self._h}].")
        else:   
            if self._E < - self._h or self._E > self._h:
                raise ValueError(f"For h >= J, the energy E must be in the interval [{-self._h}, {self._h}].")

        if self._n_samples <= 0:
            raise ValueError("The number of samples must be a positive integer.")

        IP_ic = 0
        for i, x in enumerate(self._ic):
            l_i, m_i = x[0]
            coeff_1 = x[1]
            for j, y in enumerate(self._ic):
                l_j, m_j = y[0]
                coeff_2 = y[1]
                if (l_i + m_i + l_j + m_j) % 2 == 1:
                    IP_ic += 0
                else:
                    IP_ic += self.microcananical_IP(l_i, m_i, l_j, m_j) * np.conj(coeff_1) * coeff_2
        
        new_ic = []
        for i, x in enumerate(self._ic):
            l_i, m_i = x[0]
            coeff_1 = x[1]
            new_ic.append([[l_i, m_i], coeff_1 / np.sqrt(IP_ic)])

        self._ic = new_ic

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

    def spectral_width(self) -> float:
        """
        Exact copy of the function in the classicalLMG class. See above for its description.
        """
        if self._h <= self._J:
            spectral_width =  ((self._h + self._J) ** 2) / (2 * self._J)
        else:
            spectral_width = 2 * self._h

        return spectral_width

    def microcananical_IP(self, l1: int, m1: int, l2: int, m2: int) -> float:
        """
        This fuction computes the microcanonical inner product (Y_l^m, Y_l'^m')_{E,delta_E} between two spherical harmonics Y_l^m and Y_l'^m' using Monte Carlo sampling.
        Note: the function filter_points_LMG() is called within this function to generate random points on the sphere and filter them based on the selected energy shell defined by E and delta_E. As a consequence, this Monte Carlo is statistically independent from the Monte Carlo used in the Lanczos algorithm, which is executed in the function Lanczos_coeff_MC().

        Parameters
        ----------
        l1: int
            The l number of the first spherical harmonic Y_l^m.
        
        m1: int
            The m number of the first spherical harmonic Y_l^m.

        l2: int
            The l number of the second spherical harmonic Y_l'^m'.

        m2: int
            The m number of the second spherical harmonic Y_l'^m'.

        Returns
        -------
        IP: float
            The value of the microcanonical inner product (Y_l^m, Y_l'^m')_{E,delta_E} between the two spherical harmonics Y_l^m and Y_l'^m' computed using Monte Carlo sampling.

        Example
        -------
        >>> h = 0.5
        >>> J = 1.
        >>> E = 0.
        >>> delta_E = 0.1
        >>> n_samples = 10 ** 6
        >>> LMG_MC_classical = classicalLMG_MC(h, J, [[[1, 0], 1.]], E, delta_E, n_samples)
        >>> LMG_MC_classical.microcananical_IP(1, 0, 1, 0)
        0.07068958819668766
        """
        if (l1 + l2 + m1 + m2) % 2 == 1:
            IP = 0.0
        else:
            theta, phi = filter_points_LMG(self.h, self.J, self.E, self.delta_E, self.n_samples)
            Y_product = np.conj(spc.sph_harm_y(l1, m1, theta, phi)) * spc.sph_harm_y(l2, m2, theta, phi)
            IP = np.real(np.sum(Y_product)) / len(theta)

        return IP

    def Lanczos_coeff_MC(self, b_number: int) -> np.ndarray:
        """
        Wrapper function that executes the classical microcanonical Lanczos algorithm for the classical LMG model by calling the njited function classical_MC_Lanczos_algorithm() defined in LanczosClassical.py.
        Given the number of Lanczos coefficients to be computed and the initial conditions, this method builds the initial grid containing the initial function. It also initializes an empty grid containing the vertices [l, m], both of which are required to execute the classical microcanonical Lanczos algorithm.

        Parameters
        ----------
        b_number: int
            The number of Lanczos iterations that will be performed.

        Returns
        -------
        (see LanczosClassical.classical_MC_Lanczos_algorithm())

        Example
        -------
        >>> h = 0.5
        >>> J = 1.
        >>> ic = [[[1, 0], 1.]]
        >>> E = 0.
        >>> delta_E = 0.1
        >>> n_samples = 10 ** 6
        >>> LMG_MC_classical = classicalLMG_MC(h, J, ic, E, delta_E, n_samples)
        >>> LMG_MC_classical.Lanczos_coeff_MC(30)
        array([0.18414776, 0.64994774, 0.36855749, 2.02432674, 0.49446324,
               2.69159109, 2.11955158, 1.42456902, 3.52207239, 3.14402934,
               1.87989065, 4.91640066, 3.28347014, 3.58534432, 4.54161353,
               5.57944455, 3.03453151, 6.80859407, 4.79935492, 5.70138766,
               5.46144475, 7.98552504, 4.30567107, 8.5393672 , 6.63497336,
               7.32492749, 7.14774227, 9.24850865, 7.16752373, 8.18228636])
        """
        if b_number < 0:
            raise ValueError("The number of Lanczos iterations must be a non-negative integer.")

        grid_ic, lmax = build_ic(self._ic, b_number)
        empty_grid = build_grid(lmax)

        Y_all, n_theta = precompute_all_spherical_harmonics_1(lmax, self.h, self.J, self.E, self.delta_E, self.n_samples)
        
        Y_all_R = (np.asfortranarray(Y_all), ) # we use the np.asfortranarray() function for better compatibility with njit 
        Y_all_C = (np.asfortranarray(np.conj(Y_all)), )
        
        G_mat = create_IP_mat(lmax)
        norm_MC = 1.0 / n_theta

        return classical_MC_Lanczos_algorithm(grid_ic, empty_grid, L_c, b_number, G_mat, Y_all_C, Y_all_R, norm_MC, dot_product, self.h, self.J)
    
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
        str_2 = f"Initial condition is {result}." 
        str_3 = f"The energy window is defined by E = {self.E} and delta_E = {self.delta_E}."
        
        return str_1 + " " + str_2 + " " + str_3