import numpy as np
import scipy.linalg as la
from scipy.stats import gaussian_kde

class Hamiltonian:

    def __init__(self):
        """
        Constructor for the Hamiltonian class, which serves as a base class for specific Hamiltonian models.
        """
        self._matrix: np.ndarray | None = None
        self._model_type: str | None = None

    def compute_DOS(self, bandwidth: float = 0.025, n_points: int = 600, symmetry_blocks: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        This function computes the density of states (DOS) of the Hamiltonian matrix using Gaussian kernel density estimation (KDE) to get a smooth curve.

        Parameters
        ----------
        bandwidth: float (optional)
            The bandwidth parameter for the Gaussian KDE, which controls the smoothness of the resulting DOS curve. 
            The default value is set to 0.025, which provides smooth enough DOS curves for the models considered in this work.

        n_points: int (optional)
            The number of points in the energy grid on which the DOS will be evaluated.
            The default value is set to 600, which provides a good resolution for the DOS curves of the models considered in this work.

        symmetry_blocks: bool (optional)
            Whether to compute the DOS separately for each symmetry block of the Hamiltonian. If True, the function will compute the DOS for each symmetry block and return a list of DOS curves and corresponding energy grids. 
            If False, the function will compute the DOS for the entire Hamiltonian matrix and return a single DOS curve and energy grid. The default value is False.

        Returns
        -------
        E_spectrum: numpy.ndarray
            The energy levels of the Hamiltonian matrix.

        E_grid: numpy.ndarray
            The energy grid on which the DOS is evaluated.

        rho: numpy.ndarray
            The density of states evaluated on the energy grid E_grid.

        E_symmetry_spectrum: list (only if symmetry_blocks is True)
            A tuple containing the energy levels of each symmetry block of the Hamiltonian matrix.

        E_grid_list: list (only if symmetry_blocks is True)
            A list of energy grids corresponding to each symmetry block.

        rho_list: list (only if symmetry_blocks is True)
            A list of density of states curves corresponding to each symmetry block, evaluated on the respective energy

        Example
        -------
        >>> LMG_system = LMG(0.5, 1.0, 10)
        >>> LMG_system.build_intensive()
        >>> LMG_system.compute_DOS()[0]
        array([-0.63221589, -0.63218501, -0.55838641, -0.55541005, -0.5120225 ,
               -0.48635395, -0.4474792 , -0.40449729, -0.35671567, -0.30479053,
               -0.24902993, -0.18968105, -0.12694035, -0.06097044,  0.00809079,
                0.08012428,  0.15502573,  0.23270278,  0.31307278,  0.39606127,
                0.48160064])
        """
        if self._matrix is None:
            raise ValueError("Hamiltonian matrix has not been set.")
        
        if symmetry_blocks:
            smooth_DOS_list = []
            E_grid_list = []
            rho_list = []

            E_symmetry_spectrum = self.symmetry_blocks(spectrum = True)
            n_entries = len(E_symmetry_spectrum)

            for i in range(int(n_entries / 2), n_entries):
                smooth_DOS_list.append(gaussian_kde(E_symmetry_spectrum[i], bw_method = bandwidth))
                E_grid_list.append(np.linspace(min(E_symmetry_spectrum[i]), max(E_symmetry_spectrum[i]), n_points))
                rho_list.append(smooth_DOS_list[-1](E_grid_list[-1]))

            return list(E_symmetry_spectrum[int(n_entries / 2) :]), E_grid_list, rho_list

        else:
            E_spectrum, _ = la.eigh(self.matrix)
            smooth_DOS = gaussian_kde(E_spectrum, bw_method = bandwidth)
            
            E_grid = np.linspace(min(E_spectrum), max(E_spectrum), n_points)
            rho = smooth_DOS(E_grid)

            return E_spectrum, E_grid, rho
    
    def expectation_value(self, level: int = 0, operator: np.ndarray | None = None) -> complex:
        """
        This function computes the expectation value of a given operator in the eigenstate corresponding to the specified energy level.

        Parameters
        ----------
        level: int (optional)
            The energy level for which to compute the magnetization, with 0 corresponding to the ground state. The default value is 0.

        operator: numpy.ndarray (optional)  
            The operator with respect to which to compute the magnetization. The default value is None.

        Returns
        -------
        m: complex
            The expectation value of the specified operator in the eigenstate corresponding to the specified energy level.

        Example
        -------
        >>> FP_system = FP(0.5, 10)
        >>> FP_system.build_intensive()
        >>> FP_system.magnetization()
        np.complex128(0.4111023190765097+0j)
        """
        if self._matrix is None:
            raise ValueError("Hamiltonian matrix has not been set.")
        
        _, phi_spectrum = la.eigh(self.matrix)
        m = np.conj(phi_spectrum[:, level]) @ operator @ phi_spectrum[:, level]

        return m
    
    def autocorrelation(self, operator, time_grid):
        """
        This function returns the autocorrelation function (or two-point function) of a given operator using the infinite-temperature inner product.
        The operator should not be normalized, as the function will normalize it internally to ensure that the autocorrelation function is properly normalized at time t = 0.

        Parameters
        ----------
        operator: numpy.ndarray
            The operator for which to compute the autocorrelation function.

        time_grid: numpy.ndarray
            Array of time values at which the autocorrelation function is evaluated. It is more convenient to pass the time grid as an argument to this function, and give the user more control over the time scales (e.g. linear or logarithmic) and the number of points in the time grid.

        Returns
        -------
        phi_0_sq: numpy.ndarray
            The norm squared of the autocorrelation function of the specified operator evaluated at the time values in time_grid, normalized such that phi_0_sq[0] = 1.
            
        Example
        -------
        >>> FP_system = FP(0.5, 5)
        >>> FP_system.build_intensive()
        >>> Lx, Ly, Lz = util.spin_operators(FP_system.L)
        >>> time_grid = np.linspace(0, 1, 11)
        >>> FP_system.autocorrelation(np.kron(Lx, Lx), time_grid)
        array([1.        +0.00000000e+00j, 0.99820152-4.21680019e-35j,
               0.99282428+1.02141977e-34j, 0.98392262-1.04639625e-34j,
               0.97158617-4.94760343e-35j, 0.95593846+7.98047728e-35j,
               0.93713511+3.55049945e-34j, 0.91536138-5.82806241e-35j,
               0.89082946-2.03913261e-35j, 0.86377527+7.72261507e-35j,
               0.83445497+1.90264300e-34j])
        """
        if self._matrix is None:
            raise ValueError("Hamiltonian matrix has not been set.")

        dim = self.matrix.shape[0]
        
        E_spectrum, phi_spectrum = la.eigh(self.matrix)

        op_flatten = operator.flatten()
        op_normalized = operator / np.sqrt(np.vdot(op_flatten, op_flatten) / dim)
        op_energy_basis = phi_spectrum.conj().T @ op_normalized @ phi_spectrum 

        delta_E = E_spectrum[:, None] - E_spectrum[None, :] 
        op_sq = np.abs(op_energy_basis) ** 2 

        result = np.zeros_like(time_grid, dtype = np.complex128)
        for idx, t in enumerate(time_grid):
            exp_factor_t = np.exp(1j * delta_E * t)
            result[idx] = np.sum(exp_factor_t * op_sq)

        phi_0 = result / dim

        return phi_0