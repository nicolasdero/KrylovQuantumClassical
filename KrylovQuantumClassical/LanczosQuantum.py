import numpy as np
import scipy.sparse as sp
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

class LanczosQuantum:

    def __init__(self, model: str, spin_size: int, param: list, initial_operator: list, precision: int):
        r"""
        Constructor for the KrylovQuantum class.

        Parameters
        ----------
        model: str
            The type of the quantum model, which can be either 'LMG' for the LMG model or 'FP' for the FP model.

        spin_size: int
            The spin size of the system, which must be a positive integer (or half-integer, for the LMG model).

        param: list
            A list of parameters that define the Hamiltonian of the system. 
            For the LMG model, the list must contain exactly 2 elements: [h, J], and for the FP model, the list must contain exactly 1 element: [a].

        initial_operator: list
            A list of lists that define the initial operator for the Lanczos algorithm. 
            For the LMG model, each inner list must contain exactly 3 elements: [coefficient 1, coefficient 2, coefficient 3], where the coefficients of the ith list correspond to the operators S_x^i, S_y^i, and S_z^i, respectively.
            For the FP model, each inner list must contain exactly 5 elements: [coefficient, operator 1, power 1, operator 2, power 2], which represents the operator coefficient * (operator 1)^power 1 \otimes (operator 2)^power 2, where the operators can be either S_x, S_y, or S_z. All operators constructed from each inner list of the initial_operator list will be summed together to form the initial operator for the Lanczos algorithm.

        precision: int
            Precision at whoch representing the operators in Mathematica, which must be a positive integer. This parameter controls the precision of the Lanczos algorithm, and it is important to ensure that it is set to a sufficiently high value to obtain accurate results.
        """
        self._model = model
        self._spin_size = spin_size
        self._param = param
        self._initial_operator = initial_operator
        self._precision = precision
        self._Lanczos = None
        self._a_coeff = None
        self._K_dim = None
        self._session = WolframLanguageSession()

        if self._model not in ['LMG', 'FP']:
            raise ValueError("Invalid model type. Supported models are 'LMG' and 'FP'.")
        
        if not isinstance(self._spin_size, (int, float)) or self._spin_size <= 0 or not np.isclose(self._spin_size % 1, 0):
            raise ValueError("Spin size must be a positive integer.")
        
        if self._model == 'LMG' and len(self._param) != 2:
            raise ValueError("For the LMG model, the param list must contain exactly 2 elements: [h, J].")
        
        if self._model == 'LMG' and len(self._initial_operator) >= 1:
            for x in self._initial_operator:
                if type(x) != list:
                    raise ValueError("For the LMG model, each element in the initial_operator list must be a list of 3 elements, e.g. [[1, 0, 1], [1, 1, 2], [1, 0, 1]].")
                elif len(x) != 3:
                    raise ValueError("For the LMG model, each list in the initial_operator list must be a list of 3 elements, e.g. [[1, 0, 1], [1, 1, 2], [1, 0, 1]].")
                else:
                    for y in x:
                        if not isinstance(y, (int, float)):
                            raise ValueError("For the LMG model, each element in the lists of the initial_operator list must be a float, e.g. [1.1, 0.5, 2.4].")
        if self._model == 'LMG' and len(self._initial_operator) == 0:
            raise ValueError("For the LMG model, the initial_operator list must contain at least one list of 3 elements, e.g. [[1, 0, 1]].")
        
        if self._model == 'FP' and len(self._param) != 1:
            raise ValueError("For the FP model, the param list must contain exactly 1 element: [a].")
        
        if self._model == 'FP' and len(self._initial_operator) >= 1:
            for x in self._initial_operator:
                if type(x) != list:
                    raise ValueError("For the FP model, each element in the initial_operator list must be a list of 5 elements, e.g. [[1, 1, 2, 0, 1], [1, 0, 1, 1, 2], [1, 3, 1, 3, 3]].")
                elif len(x) != 5:
                    raise ValueError("For the FP model, each list in the initial_operator list must be a list of 5 elements, e.g. [[1, 1, 2, 0, 1], [1, 0, 1, 1, 2], [1, 3, 1, 3, 3]].")
                else:
                    if not isinstance(x[0], (int, float)):
                        raise ValueError("For the FP model, the first element in the lists of the initial_operator list must be a int or float")
                    if (x[1] and x[3]) not in [0, 1, 2, 3]:
                        raise ValueError("For the FP model, the second and fourth elements in the lists of the initial_operator list must be eqal to 0, 1, 2, or 3.")
                    if not (isinstance(x[2], int) and isinstance(x[4], int) and x[2] > 0 and x[4] > 0):
                        raise ValueError("For the FP model, the second and fourth elements in the lists of the initial_operator list must be positive integers.")
                            
        elif self._model == 'FP' and len(self._initial_operator) == 0:           
            raise ValueError("For the FP model, the initial_operator list must contain at least one list of 5 elements, e.g. [[1, 1, 2, 0, 1]].")
        
        if self._model == 'FP' and np.isclose(self._spin_size % 1, 0.5):
            raise ValueError("For the FP model, the spin size L must be an integer.")
        
        if self._precision <= 0:
            raise ValueError("Precision must be a positive integer.")

    @property
    def model(self) -> str:
        return self._model
    
    @property
    def spin_size(self) -> int:
        return self._spin_size
    
    @property
    def param(self) -> list:
        return self._param

    @property
    def initial_operator(self) -> list:
        return self._initial_operator
    
    @property
    def precision(self) -> int:
        return self._precision
    
    @property   
    def Lanczos(self) -> bool:
        if self._Lanczos is None:
            raise ValueError("Lanczos coefficients have not been computed yet. Please run the Lanczos_coeff() method first.")
        return self._Lanczos

    @property
    def a_coeff(self) -> bool:
        if self._a_coeff is None:
            raise ValueError("a coefficients have not been computed yet. Please run the Lanczos_coeff_MC() method first.")
        return self._a_coeff
    
    @property
    def K_dim(self) -> int:
        if self._K_dim is None:
            raise ValueError("Lanczos coefficients have not been computed yet. Please run the Lanczos_coeff() method first.")
        return self._K_dim
    
    @property
    def session(self) -> WolframLanguageSession:
        return self._session
    
    def Lanczos_coeff_IT(self):
        """
        This function computes the Lanczos coefficients sequence using the infinite temperature Lanczos algorithm. 
        The coefficients are computed by calling a Mathematica script that implements the algorithm for the specified model and parameters. 
        The computed Lanczos coefficients are stored in the _Lanczos attribute of the class, and the dimension of the Krylov space is stored in the _K_dim attribute of the class.

        Remark 1: the Mathematica script is constructed to return the Lanczos sequence including the first (b_0) and last (b_K) zero Lanczos coefficients.
        Since the number of Lanczos coefficients is equal to K_dim - 1, the first and last zero Lanczos coefficients are removed from the returned array of Lanczos coefficients.

        Example
        -------
        >>> LMG_Lanczos_IT = KrylovQuantum(model = 'LMG', spin_size = 4, param = [0.5, 1.0], initial_operator = [[1, 0, 1]], precision = 500)
        >>> LMG_Lanczos_IT.Lanczos_coeff_IT()
        >>> LMG_Lanczos_IT.Lanczos
        array([0.125     , 0.12263386, 0.17803491, 0.36663842, 0.29933453,
               0.39983453, 0.4993746 , 0.51052699, 0.52163256, 0.59881447,
               0.61120547, 0.60946502, 0.50466394, 0.55469932, 0.31689742,
               0.46136483, 0.43429583, 0.45722714, 0.50513343, 0.43581184,
               0.46858465, 0.48140086, 0.27165137, 0.39127596, 0.30333879,
               0.23207491, 0.2327803 , 0.25081088, 0.17337794, 0.10719385,
               0.1902364 , 0.27314251, 0.42709741, 0.09318997, 0.17470567,
               0.0377407 , 0.17264257, 0.02627635, 0.13626909])
        """
        self.session.evaluate(wl.Set(wl.ic, self._initial_operator))
        self.session.evaluate(wl.Set(wl.p, self._precision))

        if self._model == 'LMG':
            self.session.evaluate(wl.Set(wl.S, self._spin_size))
            self.session.evaluate(wl.Set(wl.hVal, self._param[0]))
            self.session.evaluate(wl.Set(wl.JVal, self._param[1]))
            result = self.session.evaluate(wl.Get('LanczosAlgorithmLMG.wl'))

        if self._model == 'FP':
            self.session.evaluate(wl.Set(wl.L, self._spin_size))
            self.session.evaluate(wl.Set(wl.lambdaVal, self._param[0]))
            result = self.session.evaluate(wl.Get('LanczosAlgorithmFP.wl'))

        K_dim = result[0]
        Lanczos = np.array(result[1], dtype = float)[1 : - 1]

        if len(Lanczos) != K_dim - 1:
            raise ValueError("The number of Lanczos coefficients must be equal to K_dim - 1.")

        if len(Lanczos) != 0:
            self._Lanczos = Lanczos
            self._K_dim = K_dim

    def Lanczos_coeff_MC(self, E, E_window, one_point = False):
        """
        This function computes the Lanczos coefficients sequence using the microcanonical Lanczos algorithm.
        Similar comments as for the Lanczos_coeff_IT() method apply to the Lanczos coefficients computed by this method, 
        with the difference that the Lanczos coefficients are computed by calling a different Mathematica script that implements the microcanonical Lanczos algorithm for the specified model and parameters, 
        and that the Lanczos coefficients are computed in a specific energy window around a specified energy value. 
        The window is [E - E_window / 2, E + E_window / 2], where E is the specified energy value and E_window is the specified width of the energy window.

        Remark 1: see Lanczos_coeff_IT() method
        Remark 2: similar remark for the a coefficients, with the difference that the number of a coefficients is equal to K_dim, and thus only the last zero a coefficient is removed from the returned array of a coefficients.

        Parameters
        ----------  
        E: float
            The energy value around which to compute the microcanonical Lanczos coefficients.

        E_window: float
            The width of the energy window around the specified energy value E, within which to compute the microcanonical Lanczos coefficients.

        one_point: bool (optional)
            A boolean flag that indicates whether to remove the one-point function of the initial operator and make it orthogonal to the initial vector of the Lanczos algorithm. 
        
        Returns
        -------
        a: numpy.ndarray (of shape (self._K_dim,))
            The a coefficients of the Lanczos algorithm, which are the diagonal elements of the tridiagonal matrix representation of the Liouvillian in the Krylov basis. 

        Example
        -------
        >>> LMG_Lanczos_MC = KrylovQuantum(model = 'LMG', spin_size = 8, param = [0.5, 1.0], initial_operator = [[0, 0, 1]], precision = 500)
        >>> LMG_Lanczos_MC.Lanczos_coeff_MC(E = - 0.5, E_window = 0.2, one_point = True)
        array([ 0.00557204, -0.00310872,  0.02681525,  0.04306132,  0.04083981,
                0.13136256,  0.10459508,  0.24881463,  0.2240122 ,  0.26566577,
                0.31783675,  0.37559502,  0.45038179,  0.40837478,  0.42979684,
                0.46601324,  0.46587235,  0.46907996,  0.47857784,  0.56144294,
                0.31182091,  0.40157524,  0.40807629,  0.37629927,  0.34654384,
                0.35951985,  0.25950382,  0.22987454,  0.51995775,  0.39675116,
                0.55235666,  0.59937055,  0.35184492,  0.14993029])
        """
        self.session.evaluate(wl.Set(wl.ic, self._initial_operator))
        self.session.evaluate(wl.Set(wl.p, self._precision))
        self.session.evaluate(wl.Set(wl.energy, E))
        self.session.evaluate(wl.Set(wl.deltaE, E_window))
        if one_point:
            self.session.evaluate(wl.Set(wl.onePoint, 1))
        else:
            self.session.evaluate(wl.Set(wl.onePoint, 0))

        if self._model == 'LMG':
            if E_window <= 0:
                raise ValueError("The energy window E_window must be a positive number.")
        
            if self._param[0] <= self._param[1]:
                if E < - (self._param[0] ** 2 + self._param[1] ** 2) / (2 * self._param[1]) or E > self._param[0]:
                    raise ValueError(f"For h <= J, the energy E must be in the interval [{-(self._param[0] ** 2 + self._param[1] ** 2) / (2 * self._param[1])}, {self._param[0]}].")
            else:   
                if E < - self._param[0] or E > self._param[0]:
                    raise ValueError(f"For h >= J, the energy E must be in the interval [{- self._param[0]}, {self._param[0]}].")
                
            self.session.evaluate(wl.Set(wl.S, self._spin_size))
            self.session.evaluate(wl.Set(wl.hVal, self._param[0]))
            self.session.evaluate(wl.Set(wl.JVal, self._param[1]))
            result = self.session.evaluate(wl.Get('LanczosAlgorithmMCLMG.wl'))

        if self._model == 'FP':
            if E_window <= 0:
                raise ValueError("The energy window delta_E must be a positive number.")
        
            if self._param[0] >= -1 and self._param[0] <= 0.6:
                E_min = (17 * self._param[0] / 4) - (1 / (1 - self._param[0])) - (13 / 4)
                if E < E_min or E > - E_min:
                    raise ValueError(f"For -1 <= a <= 3/5, the energy E must be in the interval [{E_min}, {-E_min}].")
            else:
                if E < - 2 * (1 + self._param[0]) or E > 2 * (1 + self._param[0]):
                    raise ValueError(f"For 3/5 <= a <= 1, the energy E must be in the interval [{- 2 * (1 + self._param[0])}, {2 * (1 + self._param[0])}].")  

            self.session.evaluate(wl.Set(wl.L, self._spin_size))
            self.session.evaluate(wl.Set(wl.lambdaVal, self._param[0]))
            result = self.session.evaluate(wl.Get('LanczosAlgorithmMCFP.wl'))

        K_dim = result[0]
        Lanczos = np.array(result[1], dtype = float)[1 : - 1]
        a = np.array(result[2], dtype = float)[: - 1]

        if len(Lanczos) != K_dim - 1:
            raise ValueError("The number of Lanczos coefficients must be equal to K_dim - 1.")

        if len(a) != K_dim:
            raise ValueError("The number of a coefficients must be equal to K_dim.")

        if len(Lanczos) != 0:
            self._Lanczos = Lanczos
            self._K_dim = K_dim
            self._a_coeff = a
    
    def K_complexity(self, t_max, dt):
        """
        Given a Lanczos coefficients sequence (and optionally the a coefficients sequence), this function computes the K-complexity of the system as a function of time by simulating the evolution of the wavefunction in the Krylov basis under the tridiagonal matrix representation of the Liouvillian.

        Parameters
        ----------
        t_max: float
            Maximum time up to which to compute the K-complexity

        dt: float
            Time step for the evolution

        a_coeff: numpy.ndarray (of shape (self._K_dim,), optional)
            The a coefficients of the Lanczos algorithm, if any.

        Returns
        -------
        time_grid: numpy.ndarray (of shape (int(t_max / dt) + 1,))
            The grid of time points at which the K-complexity is computed, which ranges from 0 to t_max with a step of dt.

        K_C: numpy.ndarray (of shape (int(t_max / dt) + 1,))
            The K-complexity of the system at each time point in the time grid, which is computed as the expectation value of the position operator in the Krylov basis with respect to the evolved wavefunction.
            
        Example
        -------
        >>> LMG_Lanczos = KrylovQuantum(model = 'LMG', spin_size = 4, param = [0.5, 1.0], initial_operator = [[1, 0, 1]], precision = 500)
        >>> LMG_Lanczos.Lanczos_coeff_IT()
        >>> LMG_Lanczos.K_complexity(t_max = 10, dt = 0.5)[1]
        array([0.        , 0.00383548, 0.01537135, 0.03469374, 0.06193947,
               0.097287  , 0.14094577, 0.19314543, 0.25412647, 0.32413356,
               0.40341211, 0.49220851, 0.59077337, 0.69936686, 0.81826471,
               0.94776322, 1.08818146, 1.23985913, 1.40314873, 1.57840141,
               1.76594625])
        """
        if self._Lanczos is None:
            raise ValueError("Lanczos coefficients have not been computed yet. Please run the Lanczos_coeff() method first.")

        if self._a_coeff is None:
            diagonals = [self._Lanczos, self._Lanczos]
            L = sp.diags(diagonals, offsets=[- 1, 1], format='csr')

        else:
            diagonals = [self._Lanczos, self._a_coeff, self._Lanczos]
            L = sp.diags(diagonals, offsets=[- 1, 0, 1], format='csr')

        initial_vec = np.zeros(self._K_dim)
        initial_vec[0] = 1

        n_points = int(t_max / dt) + 1
        time_grid = np.linspace(0, t_max, n_points)
        K_wf = sp.linalg.expm_multiply(1j * L, initial_vec, start = 0, stop = t_max, num = n_points, endpoint = True)
        abs_K_wf = np.abs(K_wf) ** 2

        if not np.allclose(np.sum(abs_K_wf, axis = 1), 1):
            raise ValueError("The wavefunction is not normalized at all times.")
        
        n_vals = np.arange(self._K_dim)
        K_C = abs_K_wf @ n_vals 

        return time_grid, K_C
    
    def LT_K_complexity(self):
        """
        This function computes the late-time saturation value of K-complexity, see Eq. (A.4) in the main text.

        Parameters
        ----------
        a_coeff: numpy.ndarray (of shape (self._K_dim,), optional)
            The a coefficients of the Lanczos algorithm, if any.

        Returns
        -------
        LT_C_K: float
            The late-time saturation value of K-complexity.

        Q_0n: numpy.ndarray (of shape (self._K_dim,))
            The late-time average of the transition amplitude. Note that the sum of the elements of Q_0n is equal to 1.

        Example
        -------
        >>> LMG_Lanczos = KrylovQuantum(model = 'LMG', spin_size = 4, param = [0.5, 1.0], initial_operator = [[1, 0, 1]], precision = 500)
        >>> LMG_Lanczos.Lanczos_coeff_IT()
        >>> LMG_Lanczos.LT_K_complexity()[0]
        np.float64(31.513263254008578)
        """
        if self._Lanczos is None:
            raise ValueError("Lanczos coefficients have not been computed yet. Please run the Lanczos_coeff() method first.")
        
        if self._a_coeff is None:
            L = np.diag(self._Lanczos, -1) + np.diag(self._Lanczos, 1)
        else:
            L = np.diag(self._Lanczos, -1) + np.diag(self._a_coeff) + np.diag(self._Lanczos, 1)

        _, eigvec_L = np.linalg.eigh(L)

        list_n = np.arange(self._K_dim)

        Q_0n = np.zeros(self._K_dim, dtype = np.float64)

        for n in range(self._K_dim):
            transit = (np.abs(eigvec_L[0]) ** 2) * (np.abs(eigvec_L[n]) ** 2)
            Q_0n[n] = np.sum(transit)

        LT_C_K = np.dot(Q_0n, list_n)

        return LT_C_K, Q_0n
    
    def terminate(self):
        """
        This function terminates the Wolfram Language session.
        """
        self._session.terminate()
    
    def __str__(self):
        str1 = f"Krylov algorithm is set up for the {self._model} model with spin size {self._spin_size} and parameters (h, J for the LMG model and a for the FP model) {self._param}."
        str2 = f"The initial operator for the Lanczos algorithm is defined by the list {self._initial_operator}, and the precision for representing the operators in Mathematica is set to {self._precision}."

        return str1 + " " + str2
