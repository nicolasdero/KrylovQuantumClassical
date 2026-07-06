import numpy as np
import scipy.sparse as sp
from numba import njit

@njit
def extract_nonzero(grid: np.ndarray, epsilon: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    njited function that extracts the nonzero entries of a given grid, defined as the coefficients whose absolute value exceeds a given threshold.

    Parameters
    ----------
    grid: numpy.ndarray of shape (size_grid,)
        The grid from which to extract the nonzero entries, where size_grid is the total number of vertices in the grid (see its definition in classicalFP.build_grid() and classicalLMG.build_grid()).

    epsilon: float (optional)
        The threshold for the absolute value of the coefficients in the grid to be considered nonzero. The default value is 1e-12.

    Returns
    -------
    idx: numpy.ndarray of shape (n_nonzero,)
        The indices of the nonzero entries in the grid, where n_nonzero is the total number of nonzero entries in the grid.

    val: numpy.ndarray of shape (n_nonzero,)
        The coefficients associated with the nonzero entries in the grid.

    Example
    -------
    >>> grid = np.array([1 + 1j, -1e-10j, 0.5, 1e-13, 0.5e-12 + 1j])
    >>> extract_nonzero(grid)
    (array([0, 1, 2, 4]), array([ 1.e+00+1.e+00j, -0.e+00-1.e-10j,  5.e-01+0.e+00j,  5.e-13+1.e+00j]))

    Remark
    ------
    Room for future improvement: currently, extract_nonzero() performs a linear scan of the entire grid at every Lanczos iteration, traversing many vertices that remain unchanged by the application of the classical Liouvillian operator.
    Future fix: to eliminate this 'scanning the vacuum' effect, track 'active indices' during the Liouvillian application or switch to a sparse representation to avoid scanning empty grid cells.
    """
    n = grid.shape[0]
    idx = np.empty(n, np.int64)
    val = np.empty(n, dtype = np.complex128)

    count = 0
    for i in range(n):
        if np.abs(grid[i]) > epsilon:
            idx[count] = i
            val[count] = grid[i]
            count += 1

    return idx[:count], val[:count]

@njit
def classical_Lanczos_algorithm(grid_ic: np.ndarray, empty_grid: np.ndarray, L_c: callable, b_number: int, *params) -> np.ndarray:
    """
    njited function that performs the classical Lanczos algorithm for a given initial function, a given classical Liouvillian operator, and a given number of Lanczos iterations.

    Parameters
    ----------
    grid_ic: numpy.ndarray of shape (size_grid,)
        The grid of coefficients associated with the initial function, see the precise definition of the grid in classicalFP.build_ic() and classicalLMG.build_ic().

    empty_grid: numpy.ndarray of shape (size_grid, 2 or 4)
        The grid of vertices associated with the classical LMG or FP models, see the precise definition of the grid in classicalFP.build_grid() and classicalLMG.build_grid().

    L_c: function
        The classical Liouvillian operator, defined as a function that takes as input a grid of coefficients, the current vertex, its coefficient, its index on the full grid and the parameters of the model, and that updates the grid.

    b_number: int
        The number of Lanczos iterations that will be performed.

    *params: tuple
        The parameters of the model, which are passed as additional arguments to the classical Liouvillian operator.
        For the classical LMG model, the parameters are given by (h, J).
        For the classical FP model, the parameters are given by (kmax, a).

    Returns
    -------
    lanczos: numpy.ndarray of shape (b_number,)
        The array containing the Lanczos coefficients obtained from the classical Lanczos algorithm.

    Example
    -------
    (see example of usage in classicalLMG.Lanczos_coeff_IT() and classicalFP.Lanczos_coeff_IT())
    """
    grid_0 = grid_ic.copy()
    size = len(grid_ic)
    grid_1 = np.zeros(size, dtype = np.complex128)
    grid_cache = np.zeros(size, dtype = np.complex128)

    lanczos = np.empty(b_number)
    idx, val = extract_nonzero(grid_0)

    for it in range(b_number):

        grid_1[:] = 0.0

        for i in range(idx.shape[0]):
            L_c(grid_1, empty_grid[idx[i]], val[i], idx[i], *params)
        if it > 0:
            grid_1 -= lanczos[it - 1] * grid_0

        norm = np.linalg.norm(grid_1)
        lanczos[it] = norm

        if norm < 1e-15:
            lanczos = lanczos[:it + 1]
            break

        grid_1 /= norm

        if it == 0:
            grid_cache[:] = grid_1
        else:
            grid_0[:] = grid_cache
            grid_cache[:] = grid_1

        idx, val = extract_nonzero(grid_1)

    return lanczos

@njit(fastmath=True)
def IP_MC(grid_1: np.ndarray, empty_grid: np.ndarray, G_mat: np.ndarray, Y_C_tuple: tuple[np.ndarray, np.ndarray, ...], Y_R_tuple: tuple[np.ndarray, np.ndarray, ...], norm_MC: float, dot_product: callable) -> float:
    """
    njited function that performs the classical microcanonical Lanczos algorithm for a given initial function, a given classical Liouvillian operator, and a given number of Lanczos iterations. 
    The microcanonical requirements, i.e. the microcanonical inner product matrix, the precomputed spherical harmonics, and the normalization factor, are passed as arguments to the function.

    Parameters
    ----------
    grid_1: numpy.ndarray of shape (size_grid,)
        The grid of coefficients associated with the current iteration of the Lanczos algorithm.

    empty_grid: numpy.ndarray of shape (size_grid, 2 or 4)
        The grid of vertices associated with the classical LMG or FP models, see the precise definition of the grid in classicalFP.build_grid() and classicalLMG.build_grid().

    G_mat: numpy.ndarray of shape (size_grid, size_grid)
        The microcanonical inner product matrix.

    Y_C_tuple: tuple of numpy.ndarray 
        The length of the tuple is equal to the number of independent spheres in the model (1 for classical LMG, 2 for classical FP). 
        Each element of the tuple is a numpy.ndarray of shape (n_theta, grid_size), where n_theta is the number of points contained in the energy shell and grid_size is the number of vertices in the grid, containing the array of all the conjugated spherical harmonics.
        (see more details in classicalLMG.dot_product() or in classicalFP.dot_product())
    
    Y_R_tuple: tuple of numpy.ndarray of shape (n_theta, (l_max + 1) ** 2)
        The length of the tuple is equal to the number of independent spheres in the model (1 for classical LMG, 2 for classical FP). 
        Each element of the tuple is a numpy.ndarray of shape (n_theta, grid_size), where n_theta is the number of points contained in the energy shell and grid_size is the number of vertices in the grid, containing the array of all the spherical harmonics.
        (see more details in classicalLMG.dot_product() or in classicalFP.dot_product())

    norm_MC: float
        The normalization factor for the microcanonical inner product, which is equal to 1 / n_theta, where n_theta is the number of points contained in the energy shell.

    dot_product: function
        The function that computes the dot product between two spherical harmonics evaluated at the points lying within the energy shell, which is used to compute the microcanonical inner product matrix.
        (see more details in classicalLMG.dot_product() or in classicalFP.dot_product())

    Returns
    -------
    norm: float
        The value of the microcanonical norm of the grid in the current iteration of the Lanczos algorithm, computed using the microcanonical inner product matrix and the precomputed spherical harmonics.

    Example
    -------
    (see example of usage in classical_MC_Lanczos_algorithm())
    """
    idx, val = extract_nonzero(grid_1)
    n_active = len(idx)
    if n_active == 0:
        return 0.0

    for i in range(n_active):
        idx_i = idx[i]

        for j in range(i, n_active): # we make use of the symmetry of the inner product matrix
            idx_j = idx[j]

            if G_mat[idx_i, idx_j] >= 0.0: # see definition and convention for G_mat in classicalLMG.create_IP_mat() or in classicalFP.create_IP_mat()
                continue
            
            vertex_i = empty_grid[idx_i]
            vertex_j = empty_grid[idx_j]
            ip = dot_product(Y_C_tuple, Y_R_tuple, vertex_i, vertex_j)
            G_mat[idx_i, idx_j] = ip
            G_mat[idx_j, idx_i] = ip

    norm_sq = 0.0
    for i in range(n_active):
        idx_1 = idx[i]
        val_i_conj = np.conj(val[i])

        norm_sq += np.real(val_i_conj * G_mat[idx_1, idx_1] * val[i]) # diagonal part of the inner product matrix

        for j in range(i + 1, n_active):
            idx_2 = idx[j]
            norm_sq += 2.0 * np.real(val_i_conj * G_mat[idx_1, idx_2] * val[j])

    norm_sq *= norm_MC
    norm = np.sqrt(norm_sq)

    return norm

@njit
def classical_MC_Lanczos_algorithm(grid_ic: np.ndarray, empty_grid: np.ndarray, L_c: callable, b_number: int, G_mat: np.ndarray, Y_C_tuple: tuple[np.ndarray, np.ndarray, ...], Y_R_tuple: tuple[np.ndarray, np.ndarray, ...], norm_MC: float, dot_product: callable, *params) -> np.ndarray:
    """
    njited function that performs the classical microcanonical Lanczos algorithm for a given initial function, a given classical Liouvillian operator, and a given number of Lanczos iterations. 
    The microcanonical requirements, i.e. the microcanonical inner product matrix, the precomputed spherical harmonics, and the normalization factor, are passed as arguments to the function.

    Note: this function intentionally duplicates classical_Lanczos_algorithm(), differing only in the computation of the norm. Keeping the two implementations separate preserves readability and avoids unnecessary abstraction while remaining Numba-friendly.
    
    Parameters
    ----------
    grid_ic: numpy.ndarray of shape (size_grid,)
        The grid of coefficients associated with the initial function, see the precise definition of the grid in classicalFP.build_ic() and classicalLMG.build_ic().

    empty_grid: numpy.ndarray of shape (size_grid, 2 or 4)
        The grid of vertices associated with the classical LMG or FP models, see the precise definition of the grid in classicalFP.build_grid() and classicalLMG.build_grid().

    L_c: function
        The classical Liouvillian operator, defined as a function that takes as input a grid of coefficients, the current vertex, its associated coefficient, its index on the full grid and the parameters of the model, and that updates the grid.

    b_number: int
        The number of Lanczos iterations that will be performed.

    G_mat: numpy.ndarray of shape (size_grid, size_grid)
        The microcanonical inner product matrix.

    Y_C_tuple: tuple of numpy.ndarray 
        The length of the tuple is equal to the number of independent spheres in the model (1 for classical LMG, 2 for classical FP). 
        Each element of the tuple is a numpy.ndarray of shape (n_theta, grid_size), where n_theta is the number of points contained in the energy shell and grid_size is the number of vertices in the grid, containing the array of all the conjugated spherical harmonics.
        (see more details in classicalLMG.dot_product() or in classicalFP.dot_product())
    
    Y_R_tuple: tuple of numpy.ndarray of shape (n_theta, (l_max + 1) ** 2)
        The length of the tuple is equal to the number of independent spheres in the model (1 for classical LMG, 2 for classical FP). 
        Each element of the tuple is a numpy.ndarray of shape (n_theta, grid_size), where n_theta is the number of points contained in the energy shell and grid_size is the number of vertices in the grid, containing the array of all the spherical harmonics.
        (see more details in classicalLMG.dot_product() or in classicalFP.dot_product())

    norm_MC: float
        The normalization factor for the microcanonical inner product, which is equal to 1 / n_theta, where n_theta is the number of points contained in the energy shell.

    dot_product: function
        The function that computes the dot product between two spherical harmonics evaluated at the points lying within the energy shell, which is used to compute the microcanonical inner product matrix.
        (see more details in classicalLMG.dot_product() or in classicalFP.dot_product())

    *params: tuple
        The parameters of the model, which are passed as additional arguments to the classical Liouvillian operator.
        For the classical LMG model, the parameters are given by (h, J).
        For the classical FP model, the parameters are given by (kmax, a).

    Returns
    -------
    lanczos: numpy.ndarray of shape (b_number,)
        The array containing the Lanczos coefficients obtained from the classical microcanonical Lanczos algorithm.

    Example
    -------
    (see example of usage in classicalLMG.Lanczos_coeff_MC() and classicalFP.Lanczos_coeff_MC())
    """
    grid_0 = grid_ic.copy()
    size = len(grid_ic)
    grid_1 = np.zeros(size, dtype=np.complex128)
    grid_cache = np.zeros(size, dtype=np.complex128)

    lanczos = np.empty(b_number, dtype=np.float64)

    idx, val = extract_nonzero(grid_0)

    for it in range(b_number):
        grid_1[:] = 0.0

        for i in range(idx.shape[0]):
            L_c(grid_1, empty_grid[idx[i]], val[i], idx[i], *params)
            
        if it > 0:
            grid_1 -= lanczos[it - 1] * grid_0

        norm = IP_MC(grid_1, empty_grid, G_mat, Y_C_tuple, Y_R_tuple, norm_MC, dot_product)
        
        lanczos[it] = norm

        if norm < 1e-15:
            lanczos = lanczos[:it + 1]
            break

        grid_1 /= norm

        if it == 0:
            grid_cache[:] = grid_1
        else:
            grid_0[:] = grid_cache
            grid_cache[:] = grid_1

        idx, val = extract_nonzero(grid_1)

    return lanczos

def K_complexity(t_max, dt, Lanczos, check = False):
        """
        Given a Lanczos coefficients sequence, this function computes the corresponding K-complexity as a function of time by computing the evolution of the wavefunction in the Krylov basis under the tridiagonal matrix representation of the Liouvillian.
        Note 1: we use the fact that for classical systems, the a_n coefficients vanish (see a proof in the main text)
        Note 2: in the LacnzosQuantum class, the K-complexity is computed upon acting with a method on the LanczosQuantum object, while here we compute the K-complexity directly from the Lanczos coefficients sequence. This is because in the classical framework, the Lanczos coefficients are not attributes of a class, but rather computed directly from the initial function and the classical Liouvillian operator.

        Parameters
        ----------
        t_max: float
            Maximum time up to which to compute the K-complexity

        dt: float
            Time step for the evolution

        Lanczos: numpy.ndarray
            The Lanczos coefficients sequence, which is a 1D array of length equal to the number of Lanczos iterations performed.

        check: bool (optional)
            If True, checks that the wavefunction is normalized at all times. If the check fails, raises a ValueError.

        Returns
        -------
        time_grid: numpy.ndarray (of shape (int(t_max / dt) + 1,))
            The grid of time points at which the classical K-complexity is computed, which ranges from 0 to t_max with a step of dt.

        K_C: numpy.ndarray (of shape (int(t_max / dt) + 1,))
            The classical K-complexity of the system at each time point in the time grid, which is computed as the expectation value of the position operator in the Krylov basis with respect to the evolved wavefunction.
            
        Example
        -------
        >>> h = 0.5
        >>> J = 1.
        >>> ic_z = [[[1, 0], 1.]]
        >>> LMG_classical = classicalLMG(h, J, ic_z)
        >>> classical_Lanczos_LMG = LMG_classical.Lanczos_coeff_IT(250)
        >>> K_complexity(2, 0.1, classical_Lanczos_LMG, True)
        (array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. ]),
         array([0.        , 0.00249875, 0.00998005, 0.02239931, 0.03968317,
                0.06173084, 0.08841607, 0.11958965, 0.15508233, 0.19470828,
                0.23826899, 0.28555733, 0.33636198, 0.3904719 , 0.44768078,
                0.50779134, 0.57061946, 0.63599792, 0.70377981, 0.77384143,
                0.84608487]))
        """
        diagonals = [Lanczos, Lanczos]
        L = sp.diags(diagonals, offsets=[- 1, 1], format='csr')
        K_dim_truncated = len(Lanczos) + 1

        initial_vec = np.zeros(K_dim_truncated)
        initial_vec[0] = 1

        n_points = int(t_max / dt) + 1
        time_grid = np.linspace(0, t_max, n_points)
        K_wf = sp.linalg.expm_multiply(1j * L, initial_vec, start = 0, stop = t_max, num = n_points, endpoint = True)
        abs_K_wf = np.abs(K_wf) ** 2

        if check:
            if not np.allclose(np.sum(abs_K_wf, axis = 1), 1):
                raise ValueError("The wavefunction is not normalized at all times.")
        
        n_vals = np.arange(K_dim_truncated)
        K_C = abs_K_wf @ n_vals 

        return time_grid, K_C