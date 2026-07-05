from .LanczosClassical import classical_Lanczos_algorithm, classical_MC_Lanczos_algorithm

import numpy as np
from numba import njit
import spherical
import quaternionic

@njit(inline = "always")
def L_c_coefficients(l: int, m: int) -> tuple[float, float, float, float, float, float]:
    """
    njited function that defines the coefficients appearing in the definition of the classical Liouvillian operator associated with the FP model.
    Note: the function uses the fact that the coefficients of that operator are separable with respect to the pairs (l, m) and (k, n).

    Parameters
    ----------
    l: int
        The l number of the vertex.

    m: int
        The m number of the vertex.

    Returns
    -------
    (BP, BM, EP, EM, TP, TM): tuple[float, float, float, float, float, float]
        Tuple containing the coefficients.
        
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

@njit("void(complex128[:], int64[:], complex128, int64, int64, float64)", fastmath = True)
def L_c(grid: np.ndarray, vertex: np.ndarray, c: complex, index_current: int, k_max: int, a: float) -> None:
    """
    njited function that defines the action of the classical Liouvillian operator associated with the FP model on a given vertex of the grid. Its analytical expression is given by Eq. (3.44) in the main text.
    Note: the function modifies the input grid in place.

    Parameters
    ----------
    grid: numpy.ndarray of shape (size,)
        The grid on which the classical Liouvillian operator acts, where size is the total number of vertices in the grid (see its definition in build_grid() below).

    vertex: numpy.ndarray of shape (4,)
        The vertex on which the classical Liouvillian operator acts, defined by the quadruple of integers [l, m, k, n].

    c: complex
        The coefficient associated with that vertex.

    index_current: int
        The index of that vertex in the grid.

    k_max: int
        The maximum k number of the vertices in the grid.

    a: float
        The parameter a that enters in the definition of the FP model.

    Example
    -------
    (see how it is used in the function LanczosClassical.classical_Lanczos_algorithm() or in LanczosClassical.classical_MC_Lanczos_algorithm())
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
            grid[idx] += - c_al * TM_lm * BP_kn

        l1 = l + 1
        m1 = m - 1
        if abs(m1) <= l1:
            idx = vertex_index(l1, m1, k1, n1, k_block)
            grid[idx] += - c_al * EM_lm * BP_kn

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
    grid: numpy.ndarray of shape (size, 4)
        The grid of vertices, where size is the total number of vertices in the grid, given by ((l_max + 1) ** 2) * ((k_max + 1) ** 2).

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
    Given initial conditions for the classical FP model, this function builds the initial grid. The function also builds the maximum l and k numbers of the vertices in that grid which depend on the initial conditions and on the number of Lanczos iterations required.

    Parameters
    ----------
    ic: list 
        List containing the initial function decomposed in the spherical harmonics basis Y_l^m Z_k^n. The list has the structure [[[l_0, m_0, k_0, n_0], c_0], [[l_1, m_1, k_1, n_1], c_1], ...] where c_i is the coefficient associated to the vertex defined by the quadruple of integers [l_i, m_i, k_i, n_i].

    b_number: int
        The number of Lanczos iterations that will be performed.

    Returns
    -------
    grid: numpy.ndarray of shape (size,)
        The initial grid, where size is the total number of vertices in the grid, given by ((l_max + 1) ** 2) * ((k_max + 1) ** 2), where l_max and k_max are the maximum l and k numbers of the vertices in the grid.

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
            0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]), np.int64(2), np.int64(1))
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

    for i in range(n_vertices_ic):
        l, m, k, n = vertices_ic[i]
        j = vertex_index(l, m, k, n, k_block)
        grid[j] = coeffs_ic[i]

    return grid, l_max, k_max

@njit
def filter_points_FP(a: float, E: float, delta_E: float, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    njited function that generates random points on two independent spheres and filters them based on the selected energy shell defined by E and delta_E.
    The energy is given by the classical FP Hamiltonian, see Eq. (3.40) in the main text.

    Parameters
    ----------
    a:float
        The a parameter of the FP model. 

    E: float
        The energy value that defines the center of the energy shell.

    delta_E: float
        The width of the energy shell (Note: the energy shell is defined as the interval [E - delta_E / 2, E + delta_E / 2]).

    n_samples: int
        The number of random points to be generated on the sphere.

    Returns
    -------
    theta_1_filtered: numpy.ndarray
        The filtered polar angles of the points on the first sphere that lie within the energy shell.

    phi_1_filtered: numpy.ndarray
        The filtered azimuthal angles of the points on the first sphere that lie within the energy shell.

    theta_2_filtered: numpy.ndarray
        The filtered polar angles of the points on the second sphere that lie within the energy shell.

    phi_2_filtered: numpy.ndarray
        The filtered azimuthal angles of the points on the second sphere that lie within the energy shell.

    Example
    -------
    >>> a = 0.5
    >>> E = 0.
    >>> delta_E = 0.1
    >>> n_samples = 10 ** 2
    >>> filter_points_FP(a, E, delta_E, n_samples)
    (array([0.717399  , 1.15697124, 2.2443017 , 2.60057844, 2.06039503,
        1.10995783, 2.76619068]),
     array([4.11990781, 4.40613021, 5.49559289, 2.14966199, 1.61480355,
        0.57181791, 1.94078367]),
     array([1.80997926, 1.86850306, 1.08816039, 0.50049511, 1.14444796,
        1.6005232 , 0.24028026]),
     array([0.06158231, 5.10545084, 5.02910089, 1.08686358, 1.6019796 ,
        4.30182836, 0.71700674]))
    """
    cos_theta_1 = np.random.uniform(-1.0, 1.0, size = n_samples)
    cos_theta_2 = np.random.uniform(-1.0, 1.0, size = n_samples)
    theta_1 = np.arccos(cos_theta_1)
    theta_2 = np.arccos(cos_theta_2)
    phi_1 = np.random.uniform(0, 2 * np.pi, size = n_samples)
    phi_2 = np.random.uniform(0, 2 * np.pi, size = n_samples)

    H = (1 + a) * (np.cos(theta_1) + np.cos(theta_2)) + 4 * (1 - a) * np.sin(theta_1) * np.sin(theta_2) * np.cos(phi_1) * np.cos(phi_2)
    mask = np.abs(H - E) < delta_E / 2

    theta_1_filtered = theta_1[mask]
    phi_1_filtered = phi_1[mask]
    theta_2_filtered = theta_2[mask]
    phi_2_filtered = phi_2[mask]

    return theta_1_filtered, phi_1_filtered, theta_2_filtered, phi_2_filtered

def precompute_all_spherical_harmonics_2(l_max: int, k_max: int, a: float, E: float, delta_E: float, n_samples: int) -> tuple[np.ndarray, np.ndarray, int, int]: 
    """
    Given the maximum l and k numbers of both sets of spherical harmonics, the parameters of the classical FP model, and the energy shell defined by E and delta_E, this function precomputes all the spherical harmonics on both spheres for all the points lying within the selected energy shell on the two spheres. (sampled using the function filter_points_FP()).
    Note that the ordering of the spherical harmonics is given by the ordering of the vertices in the grid, see build_grid() above.

    Parameters
    ---------- 
    l_max: int
        The maximum l number of the spherical harmonics.

    k_max: int
        The maximum k number of the spherical harmonics.
    
    a: float
        The a parameter of the FP model.

    E: float
        The energy value that defines the center of the energy shell.

    delta_E: float
        The width of the energy shell (Note: the energy shell is defined as the interval [E - delta_E / 2, E + delta_E / 2]).

    n_samples: int
        The number of random points to be generated on the sphere.

    Returns
    -------
    Y_all_1: numpy.ndarray of shape (n_theta, (l_max + 1) ** 2)
        The array containing all the spherical harmonics evaluated at the points of the first sphere lying within the energy shell, where n_theta_1 is the number of points contained in the shell.

    Y_all_2: numpy.ndarray of shape (n_theta, (k_max + 1) ** 2)
        The array containing all the spherical harmonics evaluated at the points of the second sphere lying within the energy shell, where n_theta_2 is the number of points contained in the shell.

    n_theta_1: int
        The number of points contained in the first energy shell.

    n_theta_2: int
        The number of points contained in the second energy shell.

    Example
    -------
    >>> l_max = 1
    >>> k_max = 1
    >>> a = 0.5
    >>> E = 0.
    >>> delta_E = 0.1
    >>> n_samples = 10 ** 2
    >>> precompute_all_spherical_harmonics_2(l_max, k_max, a, E, delta_E, n_samples)[1]
    array([[ 0.28209479+0.j,  0.21665315-0.03190372j, -0.37791473+0.j, -0.21665315-0.03190372j],
           [ 0.28209479+0.j,  0.27745191-0.17338505j,  0.15699855+0.j, -0.27745191-0.17338505j],
           [ 0.28209479+0.j, -0.10530046-0.17967933j, -0.38985474+0.j,  0.10530046-0.17967933j],
           [ 0.28209479+0.j, -0.168637  -0.27660926j, -0.1697945 +0.j,  0.168637  -0.27660926j],
           [ 0.28209479+0.j,  0.10810111+0.30051883j, -0.18638019+0.j, -0.10810111+0.30051883j],
           [ 0.28209479+0.j, -0.02914013+0.07516841j,  0.47511424+0.j,  0.02914013+0.07516841j],
           [ 0.28209479+0.j,  0.19656712-0.0049934j , -0.40175276+0.j, -0.19656712-0.0049934j]])
    """
    theta_1, phi_1, theta_2, phi_2 = filter_points_FP(a, E, delta_E, n_samples)

    R_1 = quaternionic.array.from_spherical_coordinates(theta_1, phi_1) 
    wigner_1 = spherical.Wigner(l_max) 
    Y_all_1 = wigner_1.sYlm(0, R_1) 

    R_2 = quaternionic.array.from_spherical_coordinates(theta_2, phi_2)
    wigner_2 = spherical.Wigner(k_max)
    Y_all_2 = wigner_2.sYlm(0, R_2)

    n_theta_1 = len(theta_1)
    n_theta_2 = len(theta_2)

    return Y_all_1, Y_all_2, n_theta_1, n_theta_2

@njit(fastmath=True)
def create_IP_mat(empty_grid: np.ndarray) -> np.ndarray:
    """
    Function that creates the microcanonical inner product matrix for the classical FP model. The shape of the matrix corresponds to the number of vertices in the grid.
    Note that at this stage, the matrix is initialized with −1, except for entries forbidden by the parity selection rule, which are initialized to 0. It is later filled with the microcanonical inner products (Y_l^m Z_k^n, Y_l'^m' Z_k'^n')_{E,delta_E}, at position (vertex_index(l, m, k, n, (k_max + 1) ** 2), vertex_index(l', m', k', n', (k_max + 1) ** 2)) in the matrix. 
    The inner product is zero if the parity selection rule is not satisfied, i.e., if l + m + k + n + l' + m' + k' + n' is odd (not proved in the main text).

    Parameters
    ----------
    empty_grid: numpy.ndarray of shape (size, 4)
        The grid of vertices, where size is the total number of vertices in the grid, given by ((l_max + 1) ** 2) * ((k_max + 1) ** 2).

    Returns
    -------
    G: numpy.ndarray of shape ((l_max + 1) ** 2 * (k_max + 1) ** 2, (l_max + 1) ** 2 * (k_max + 1) ** 2)
        The microcanonical inner product matrix for the classical FP model.

    Example
    -------
    >>> empty_grid = build_grid(0, 1)
    >>> create_IP_mat(empty_grid)
    array([[-1., -1.,  0., -1.],
           [-1., -1.,  0., -1.],
           [ 0.,  0., -1.,  0.],
           [-1., -1.,  0., -1.]])
    """
    size = empty_grid.shape[0]
    G = np.full((size, size), -1.0)
    
    for i in range(size):
        l_i, m_i, k_i, n_i = empty_grid[i]
        
        for j in range(size):
            l_j, m_j, k_j, n_j = empty_grid[j]

            if (l_i + m_i + k_i + n_i + l_j + m_j + k_j + n_j) % 2 == 1:
                G[i, j] = 0.0
                G[j, i] = 0.0
                
    return G

@njit(fastmath=True)
def dot_product(Y_C_tuple: tuple[np.ndarray, np.ndarray], Y_R_tuple: tuple[np.ndarray, np.ndarray], vertex_i: np.ndarray, vertex_j: np.ndarray) -> float:
    """
    njited function that computes the unnormalized microcanonical inner product (Y_l^m Z_k^n, Y_l'^m' Z_k'^n')_{E,delta_E} between the product of spherical harmonics Y_l^m * Z_k^n  and Y_l'^m' * Z_k'^n' using Monte Carlo sampling. The inner product is computed as a sum over the points in the energy shell.
    The normalization factor is included later in LanczosClassical.IP_MC().

    Parameters
    ----------  
    Y_C_tuple: tuple[numpy.ndarray, numpy.ndarray]
    Tuple containing the conjugated spherical harmonics evaluated at the sampled points in the energy shell.
        - Y_C_tuple[0] has shape (n_theta, (l_max + 1) ** 2) and contains the conjugated spherical harmonics on the first sphere.
        - Y_C_tuple[1] has shape (n_theta, (k_max + 1) ** 2) and contains the conjugated spherical harmonics on the second sphere.

    Y_R_tuple: tuple[numpy.ndarray, numpy.ndarray]
    Tuple containing the spherical harmonics evaluated at the sampled points in the energy shell.
        - Y_R_tuple[0] has shape (n_theta, (l_max + 1) ** 2) and contains the spherical harmonics on the first sphere.
        - Y_R_tuple[1] has shape (n_theta, (k_max + 1) ** 2) and contains the spherical harmonics on the second sphere.

    vertex_i: numpy.ndarray of shape (4,)
        Active vertex in the grid corresponding to [l, m, k, n]. An active vertex corresponds to a vertex over which the classical Liouvillian operator acts.

    vertex_j: numpy.ndarray of shape (4,)
        Active vertex in the grid corresponding to [l', m', k', n']. An active vertex corresponds to a vertex over which the classical Liouvillian operator acts.

    Returns
    -------
    result_dot_product: float
        The value of the unnormalized microcanonical inner product (Y_l^m Z_k^n, Y_l'^m' Z_k'^n')_{E,delta_E} between the two spherical harmonics Y_l^m * Z_k^n  and Y_l'^m' * Z_k'^n' computed using Monte Carlo sampling.

    Example
    -------
    >>> l_max = 1
    >>> k_max = 1
    >>> k_block = (k_max + 1) ** 2
    >>> a = 0.5
    >>> E = 0.
    >>> delta_E = 0.1
    >>> n_samples = 10 ** 2
    >>> Y_all_1, Y_all_2, n_theta_1, n_theta_2 = KQC.classicalFP.precompute_all_spherical_harmonics_2(l_max, k_max, a, E, delta_E, n_samples)
    >>> vertex_i = [1, 0, 1, 0]
    >>> vertex_j = [1, 1, 1, 1]
    >>> dot_product((np.conj(Y_all_1), np.conj(Y_all_2)), (Y_all_1, Y_all_2), vertex_i, vertex_j)
    0.00096151985726235
    """
    l_i, m_i, k_i, n_i = vertex_i[0], vertex_i[1], vertex_i[2], vertex_i[3]
    l_j, m_j, k_j, n_j = vertex_j[0], vertex_j[1], vertex_j[2], vertex_j[3]
    
    if (l_i + m_i + k_i + n_i + l_j + m_j + k_j + n_j) % 2 == 1:
        return 0.0

    Y1_C, Y2_C = Y_C_tuple
    Y1_R, Y2_R = Y_R_tuple
    n_samples = Y1_C.shape[0]
    accum = 0.0 + 0.0j

    flat_i1 = l_i * (l_i + 1) + m_i
    flat_i2 = k_i * (k_i + 1) + n_i 
    
    flat_j1 = l_j * (l_j + 1) + m_j
    flat_j2 = k_j * (k_j + 1) + n_j
    
    for s in range(n_samples):
        val_i_c = Y1_C[s, flat_i1] * Y2_C[s, flat_i2]
        val_j_r = Y1_R[s, flat_j1] * Y2_R[s, flat_j2]
        accum += val_i_c * val_j_r

    result_dot_product = np.real(accum)
        
    return result_dot_product


class classicalFP():

    def __init__(self, a: float, ic: list):
        """
        Constructor for the classical FP class.

        Parameters
        ----------
        a: float
            The parameter a that enters in the definition of the FP model.

        ic: list
            List containing the initial function (see the precise structure of the list in the function build_ic() above).
        """
        self._a = a
        self._ic = ic

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
                if l < 0 or k < 0:
                    raise ValueError("The l and k numbers of each vertex in the initial conditions list must be non-negative integers.")
                if abs(m) > l:
                    raise ValueError("The m number of each vertex in the initial conditions list must satisfy -l <= m <= l.")
                if abs(n) > k:
                    raise ValueError("The n number of each vertex in the initial conditions list must satisfy -k <= n <= k.")
                if not isinstance(coeff, (int, float, complex)):
                    raise ValueError("The coefficient in each element of the initial conditions list must be a number.")

    @property
    def a(self):
        return self._a

    @property
    def ic(self):
        return self._ic

    def Lanczos_coeff_IT(self, b_number: int) -> np.ndarray:
        """
        Wrapper function that executes the classical Lanczos algorithm for the classical FP model by calling the njited function classical_Lanczos_algorithm() defined in LanczosClassical.py.
        Given the number of Lanczos coefficients to be computed and the initial conditions, this method builds the initial grid containing the initial function. It also initializes an empty grid containing the vertices [l, m, k, n], both of which are required to execute the classical Lanczos algorithm.

        Parameters
        ----------
        b_number: int
            The number of Lanczos iterations that will be performed.

        Returns
        -------
        (see LanczosClassical.classical_Lanczos_algorithm())

        Example
        -------
        >>> a = 0.
        >>> ic_zz = [[[1, 0, 1, 0], 1.]]
        >>> FP_classical = classicalFP(a, ic_zz)
        >>> FP_classical.Lanczos_coeff_IT(20)
        array([ 2.52982213,  3.47233968,  4.72013411,  5.66685709,  6.21286043,
                7.44283249,  7.4952765 ,  9.58306377,  9.4105079 , 11.11484156,
                11.3871763 , 12.48841138, 13.29722433, 14.09188443, 15.1276915 ,
                15.82689119, 16.80241057, 17.59512367, 18.44895755, 19.39593244])
        """
        if b_number < 0:
            raise ValueError("The number of Lanczos iterations must be a non-negative integer.")
        
        grid_ic, lmax, kmax = build_ic(self._ic, b_number)
        empty_grid = build_grid(lmax, kmax)
        return classical_Lanczos_algorithm(grid_ic, empty_grid, L_c, b_number, kmax, self.a)
    
    def __str__(self):
        terms = []

        for (l, m, k, n), coeff in self._ic:
            term = f"Y_{l}^{m} Z_{k}^{n}"
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

        str_1 = f"Classical FP model with parameter a = {self.a}." 
        str_2 = f"Initial condition is {result}." 
        
        return str_1 + " " + str_2

class classicalFP_MC():

    def __init__(self, a: float, ic: list, E: float, delta_E: float, n_samples: int):
        """
        Constructor for the classical and microcanonical LMG class.

        Parameters
        ----------
        a: float
            The parameter a that enters in the definition of the FP model.

        ic: list
            List containing the initial function (see the precise structure of the list in the function build_ic() above).

        E: float
            The energy value that defines the center of the energy shell.

        delta_E: float
            The width of the energy shell (Note: the energy shell is defined as the interval [E - delta_E / 2, E + delta_E / 2]).

        n_samples: int
            The number of random points to be generated on the sphere.
        """
        self._a = a
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
                    raise ValueError("Each element of the initial conditions list must be a list of the form [[l, m, k, n], c].")
                vertex, coeff = element
                if not isinstance(vertex, list) or len(vertex) != 4:
                    raise ValueError("The vertex in each element of the initial conditions list must be a list of four integers [l, m, k, n].")
                l, m, k, n = vertex
                if not all(isinstance(x, int) for x in vertex):
                    raise ValueError("The vertex in each element of the initial conditions list must be a list of four integers [l, m, k, n].")
                if l < 0 or k < 0:
                    raise ValueError("The l and k numbers of each vertex in the initial conditions list must be non-negative integers.")
                if abs(m) > l:
                    raise ValueError("The m number of each vertex in the initial conditions list must satisfy -l <= m <= l.")
                if abs(n) > k:
                    raise ValueError("The n number of each vertex in the initial conditions list must satisfy -k <= n <= k.")
                if not isinstance(coeff, (int, float, complex)):
                    raise ValueError("The coefficient in each element of the initial conditions list must be a number.")
                
        if self._delta_E <= 0:
            raise ValueError("The energy window delta_E must be a positive number.")
        
        if self._a >= -1 and self._a <= 0.6:
            E_min = (17 * self._a / 4) - (1 / (1 - self._a)) - (13 / 4)
            if self._E < E_min or self._E > - E_min:
                raise ValueError(f"For -1 <= a <= 3/5, the energy E must be in the interval [{E_min}, {-E_min}].")
        else:
            if self._E < - 2 * (1 + self._a) or self._E > 2 * (1 + self._a):
                raise ValueError(f"For 3/5 <= a <= 1, the energy E must be in the interval [{- 2 * (1 + self._a)}, {2 * (1 + self._a)}].")  

        if self._n_samples <= 0:
            raise ValueError("The number of samples must be a positive integer.")

    @property
    def a(self):
        return self._a

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

    def Lanczos_coeff_MC(self, b_number: int) -> np.ndarray:
        """
        Wrapper function that executes the classical microcanonical Lanczos algorithm for the classical FP model by calling the njited function classical_MC_Lanczos_algorithm() defined in LanczosClassical.py.
        Given the number of Lanczos coefficients to be computed and the initial conditions, this method builds the initial grid containing the initial function. It also initializes an empty grid containing the vertices [l, m, k, n], both of which are required to execute the classical microcanonical Lanczos algorithm.

        Parameters
        ----------
        b_number: int
            The number of Lanczos iterations that will be performed.

        Returns
        -------
        (see LanczosClassical.classical_MC_Lanczos_algorithm())

        Example
        -------
        >>> a = 0.5
        >>> ic = [[[1, 0, 1, 0], 1]]
        >>> E = 0.
        >>> delta_E = 0.1
        >>> n_samples = 10 ** 5
        >>> FP_MC_classical = classicalFP_MC(a, ic, E, delta_E, n_samples)
        >>> FP_MC_classical.Lanczos_coeff_MC(10)
        array([0.10014973, 2.41098804, 4.69254532, 5.39095384, 3.55756995, 7.25053814, 6.29358939, 6.99912847, 7.73005671, 8.74035425])
        """
        if b_number < 0:
            raise ValueError("The number of Lanczos iterations must be a non-negative integer.")

        grid_ic, lmax, kmax = build_ic(self._ic, b_number)
        empty_grid = build_grid(lmax, kmax)

        Y_all_1, Y_all_2, n_theta, _ = precompute_all_spherical_harmonics_2(lmax, kmax, self.a, self.E, self.delta_E, self.n_samples)
        
        Y_all_R = (np.asfortranarray(Y_all_1), np.asfortranarray(Y_all_2)) # we use the np.asfortranarray() function for better compatibility with njit 
        Y_all_C = (np.asfortranarray(np.conj(Y_all_1)), np.asfortranarray(np.conj(Y_all_2)))
        
        G_mat = create_IP_mat(empty_grid)
        norm_MC = 1.0 / n_theta

        return classical_MC_Lanczos_algorithm(grid_ic, empty_grid, L_c, b_number, G_mat, Y_all_C, Y_all_R, norm_MC, dot_product, kmax, self.a)
    
    def __str__(self):
        terms = []

        for (l, m, k, n), coeff in self._ic:
            term = f"Y_{l}^{m} Z_{k}^{n}"
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

        str_1 = f"Classical FP model with parameter a = {self.a}." 
        str_2 = f"Initial condition is {result}." 
        str_3 = f"The energy window is defined by E = {self.E} and delta_E = {self.delta_E}."
        
        return str_1 + " " + str_2 + " " + str_3