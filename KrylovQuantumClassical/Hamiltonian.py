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

    def compute_DOE(self, bandwidth: float = 0.025, n_points: int = 600):
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

        Returns
        -------
        E_spectrum: numpy.ndarray
            The energy levels of the Hamiltonian matrix.

        E_grid: numpy.ndarray
            The energy grid on which the DOE is evaluated.

        rho: numpy.ndarray
            The density of states evaluated on the energy grid E_grid.

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
        if self._matrix is None:
            raise ValueError("Hamiltonian matrix has not been set.")
        
        E_spectrum, phi_spectrum = la.eigh(self.matrix)
        smooth_DOE = gaussian_kde(E_spectrum, bw_method = bandwidth) # The bandwidth is set to 0.025, which is a value that provides smooth enough DOE curves
        
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
        """

        if self._matrix is None:
            raise ValueError("Hamiltonian matrix has not been set.")
        
        _, phi_spectrum = la.eigh(self.matrix)
        m = np.conj(phi_spectrum[:, level]) @ operator @ phi_spectrum[:, level]

        return m

    

    # provide example in compute_DOE !!!!