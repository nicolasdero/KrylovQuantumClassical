import numpy as np
from numba import njit

@njit
def extract_nonzero(grid, epsilon = 1e-12):
    """
    njited function that extracts the nonzero entries of a given grid, defined as the set of all the coefficients in the grid with absolute value is greater than a given threshold.

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
        The coefficients associated to the nonzero entries in the grid.

    Example
    -------
    >>> grid = np.array([1 + 1j, -1e-10j, 0.5, 1e-13, 0.5e-12 + 1j])
    >>> extract_nonzero(grid)
    (array([0, 1, 2, 4]), array([ 1.e+00+1.e+00j, -0.e+00-1.e-10j,  5.e-01+0.e+00j,  5.e-13+1.e+00j]))
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
def classical_Lanczos_algorithm_njit(grid_ic, grid_empty, L_c, b_number, *params):
    """
    njited function that performs the classical Lanczos algorithm for a given initial function, a given classical Liouvillian operator, and a given number of Lanczos iterations.

    Parameters
    ----------
    grid_ic: numpy.ndarray of shape (size_grid,)
        The grid of coefficients associated to the initial function, see the precise definition of the grid in classicalFP.build_ic() and classicalLMG.build_ic().

    grid_empty: numpy.ndarray of shape (size_grid, 2 or 4)
        The grid of vertices associated to the classical LMG or FP models, see the precise definition of the grid in classicalFP.build_grid() and classicalLMG.build_grid().

    L_c: function
        The classical Liouvillian operator, defined as a function that takes as input a grid of coefficients, the current vertex, its associated coefficient, its index on the full grid and the parameters of the model, and that updates the grid.

    b_number: int
        The number of Lanczos iterations that will be performed.

    *params: tuple
        The parameters of the model, which are passed as additional arguments to the classical Liouvillian operator.
        For th classical LMG model, the parameters are given by (h, J).
        For the classical FP model, the parameters are given by (kmax, a).

    Returns
    -------
    lanczos: numpy.ndarray of shape (b_number,)
        The array containing the Lanczos coefficients obtained from the classical Lanczos algorithm.

    Example
    -------
    (see example of usage in classicalLMG.classical_Lanczos_algorithm() and classicalFP.classical_Lanczos_algorithm())
    """

    grid_0 = grid_ic
    size = len(grid_ic)
    grid_1 = np.zeros(size, dtype=np.complex128)
    grid_cache = np.zeros(size, dtype=np.complex128)

    lanczos = np.empty(b_number)
    idx, val = extract_nonzero(grid_0)

    for it in range(b_number):

        grid_1[:] = 0.0

        for i in range(idx.shape[0]):
            L_c(grid_1, grid_empty[idx[i]], val[i], idx[i], *params)
        if it > 0:
            grid_1 -= lanczos[it - 1] * grid_0

        norm = np.linalg.norm(grid_1)
        lanczos[it] = norm

        grid_1 /= norm

        if it == 0:
            grid_cache[:] = grid_1
        else:
            grid_0[:] = grid_cache
            grid_cache[:] = grid_1

        idx, val = extract_nonzero(grid_1)

    return lanczos