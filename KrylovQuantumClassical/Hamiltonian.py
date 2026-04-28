import numpy as np
import numpy.linalg as la
from scipy.stats import gaussian_kde

class Hamiltonian:

    def __init__(self):
        """
        Constructor for the Hamiltonian class, which serves as a base class for specific Hamiltonian models.
        """
        self._matrix: np.ndarray | None = None
        self._model_type: str | None = None

    def compute_DOE(self, bandwidth: float = 0.025, n_points: int = 600, symmetry_blocks: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        This function computes the density of states (DOE) of the Hamiltonian matrix using Gaussian kernel density estimation (KDE) to get a smooth curve.

        Parameters
        ----------
        bandwidth: float (optional)
            The bandwidth parameter for the Gaussian KDE, which controls the smoothness of the resulting DOE curve. 
            The default value is set to 0.025, which provides smooth enough DOE curves for the models considered in this work.

        n_points: int (optional)
            The number of points in the energy grid on which the DOE will be evaluated.
            The default value is set to 600, which provides a good resolution for the DOE curves of the models considered in this work.

        symmetry_blocks: bool (optional)
            Whether to compute the DOE separately for each symmetry block of the Hamiltonian. If True, the function will compute the DOE for each symmetry block and return a list of DOE curves and corresponding energy grids. 
            If False, the function will compute the DOE for the entire Hamiltonian matrix and return a single DOE curve and energy grid. The default value is False.

        Returns
        -------
        E_spectrum: numpy.ndarray
            The energy levels of the Hamiltonian matrix.

        E_grid: numpy.ndarray
            The energy grid on which the DOE is evaluated.

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
        >>> LMG_system.compute_DOE()[0]
        array([-0.63221589, -0.63218501, -0.55838641, -0.55541005, -0.5120225 ,
               -0.48635395, -0.4474792 , -0.40449729, -0.35671567, -0.30479053,
               -0.24902993, -0.18968105, -0.12694035, -0.06097044,  0.00809079,
                0.08012428,  0.15502573,  0.23270278,  0.31307278,  0.39606127,
                0.48160064])
        """
        if symmetry_blocks:
            smooth_DOE_list = []
            E_grid_list = []
            rho_list = []

            E_symmetry_spectrum = self.symmetry_blocks(spectrum = True)
            n_entries = len(E_symmetry_spectrum)

            for i in range(int(n_entries / 2), n_entries):
                smooth_DOE_list.append(gaussian_kde(E_symmetry_spectrum[i], bw_method = bandwidth))
                E_grid_list.append(np.linspace(min(E_symmetry_spectrum[i]), max(E_symmetry_spectrum[i]), n_points))
                rho_list.append(smooth_DOE_list[-1](E_grid_list[-1]))

            return list(E_symmetry_spectrum[int(n_entries / 2) :]), E_grid_list, rho_list

        else:
            E_spectrum, _ = la.eigh(self.matrix)
            smooth_DOE = gaussian_kde(E_spectrum, bw_method = bandwidth)
            
            E_grid = np.linspace(min(E_spectrum), max(E_spectrum), n_points)
            rho = smooth_DOE(E_grid)

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