from .Hamiltonian import Hamiltonian
from .util import spin_operators

import numpy as np
import scipy.linalg as la


class FP(Hamiltonian):

    def __init__(self, a: float, L: float, E: float = 1):
        """
        Constructor for the FP class, which inherits from the Hamiltonian class.

        Parameters
        ----------
        a: float
            FP model parameter that controls the relative strength of the two terms in the Hamiltonian
        
        L: int (L = 1, 2, 3, ...)
            The angular momentum quantum number (L controls both collective spin sizes in the FP model, as we work in the Hilbert space of two collective spins of size L)

        E: float (optional)
            An overall energy scale that can be set to 1 without loss of generality
        """
        super().__init__()
        self._a = a
        self._E = E

        if L <= 0 or not np.isclose(L, round(L)):
            raise ValueError("L must be a positive integer or half-integer.")
        else:
            self._L = L

    @property
    def a(self) -> float:
        return self._a

    @property
    def L(self) -> float:
        return self._L

    @property
    def E(self) -> float:
        return self._E
    
    @property
    def matrix(self) -> np.ndarray | None:
        if self._matrix is None:
            raise ValueError("Hamiltonian matrix has not been set.")
        return self._matrix

    def build_extensive(self) -> np.ndarray:
        """
        The function constructs the extensive Feingold-Peres (FP) model Hamiltonian, see Eq. (2.7) in the main text.

        Returns
        -------
        H: numpy.ndarray
            The extensive Hamiltonian of the FP model in the spin-(L, L) representation, which is a (2S + 1)^2 x (2S + 1)^2 matrix
            
        Example
        -------
        >>> FP_system = FP(0.5, 0.5)
        >>> FP_system.build_extensive()
        array([[-1.5+0.j, -0. +0.j, -0. +0.j, -1. +0.j],
               [-0. +0.j, -0. +0.j, -1. +0.j, -0. +0.j],
               [-0. +0.j, -1. +0.j, -0. +0.j, -0. +0.j],
               [-1. +0.j, -0. +0.j, -0. +0.j,  1.5+0.j]])
        """
        Lx, _, Lz = spin_operators(self._L)
        D = int(2 * self._L + 1)
        
        H = - self._E * (1 + self._a) * (np.kron(Lz, np.eye(D)) + np.kron(np.eye(D), Lz)) - (4 * self._E * (1 - self._a) / self._L) * (np.kron(Lx, Lx))

        self._matrix = H
        self._model_type = "extensive"
        
        return H

    def build_intensive(self) -> np.ndarray:
        """
        The function constructs the intensive FP model Hamiltonian, see Eqs. (2.8) in the main text.

        Returns
        -------
        H: numpy.ndarray
            The intensive Hamiltonian of the FP model in the spin-(L, L) representation, which is a (2S + 1) x (2S + 1) matrix
            
        Example
        -------
        >>> FP_system = FP(0.5, 0.5)
        >>> FP_system.build_intensive()
        array([[-3.+0.j, -0.+0.j, -0.+0.j, -2.+0.j],
               [-0.+0.j, -0.+0.j, -2.+0.j, -0.+0.j],
               [-0.+0.j, -2.+0.j, -0.+0.j, -0.+0.j],
               [-2.+0.j, -0.+0.j, -0.+0.j,  3.+0.j]])
        """
        Lx, _, Lz = spin_operators(self._L)
        D = int(2 * self._L + 1)

        H = - (self._E * (1 + self._a) / self._L) * (np.kron(Lz, np.eye(D)) + np.kron(np.eye(D), Lz)) - (4 * self._E * (1 - self._a) / self._L ** 2) * (np.kron(Lx, Lx))

        self._matrix = H
        self._model_type = "intensive"
        
        return H
    
    def spectral_width(self) -> float:
        """
        This function computes the spectral width of the intensive FP Hamiltonian, which is defined as the difference between the maximum and minimum eigenvalues of the Hamiltonian.
        The formula is given by Eq. (C.17) in the main text and is computed from the corresponding classical Hamilonian.

        Returns
        -------
        width: float
            The spectral width of the intensive FP Hamiltonian.

        Example
        -------
        >>> FP_system = FP(0.8, 1)
        >>> FP_system.spectral_width()  
        7.2
        """
        if - 1. <= self._a < 0.6:
            width = - (17 * self._a / 2) + 2 / (1 - self._a) + 13 / 2
            return width

        elif 0.6 <= self._a <= 1.:
            width = 4. * (1 + self._a)
            return width

        else:
            raise ValueError("a must be in the range [-1, 1].")

    def magnetization(self, level: int = 0, operator: np.ndarray | None = None) -> complex:
        """
        This function computes the magnetization of the eigenstate corresponding to the specified energy level, with respect to the specified operator. If no operator is provided, it defaults to the tensor product of the rescaled x-component of the collective spin operator which is the order parameter of the FP model.

        Parameters
        ----------
        level: int (optional)
            The energy level for which to compute the magnetization, with 0 corresponding to the ground state. The default value is 0.

        operator: numpy.ndarray (optional)  
            The operator with respect to which to compute the magnetization. The default value is None.

        Returns
        -------
        m: complex
            The magnetization of the eigenstate corresponding to the specified energy level, with respect to the specified operator.

        Example
        -------
        See Hamitlonian.expectation_value() for an example of how to use this function.
        """
        if operator is None:
            Sx, _, _ = spin_operators(self._L)
            operator = np.kron(Sx, Sx) / self._L ** 2

        return self.expectation_value(level, operator)
    
    def blocks_dimension(self) -> tuple[int, int, int, int]:
        """
        This function computes the dimensions of the four parity blocks of the FP Hamiltonian in the spin-(L, L) representation (L must be integer), which are labeled by the eigenvalues of the parity and exchange operators. 
        The dimensions of the blocks are given by Eq. (C.3) in the main text.

        Returns
        -------
        dim_p_p: int
            Dimension of the (+, +) parity block of the FP Hamiltonian.

        dim_p_m: int
            Dimension of the (+, -) parity block of the FP Hamiltonian.

        dim_m_p: int
            Dimension of the (-, +) parity block of the FP Hamiltonian.

        dim_m_m: int
            Dimension of the (-, -) parity block of the FP Hamiltonian.

        Example
        -------
        >>> FP_system = FP(0.5, 1)
        >>> FP_system.blocks_dimension()
        (4, 1, 2, 2)
        """
        dim_p_p = (self.L + 1) ** 2
        dim_p_m = self.L ** 2
        dim_m_p = self.L * (self.L + 1)
        dim_m_m = dim_m_p

        return dim_p_p, dim_p_m, dim_m_p, dim_m_m

    def parity_operator(self) -> np.ndarray:
        r"""
        This function constructs the parity operator of the FP model in the spin-(L, L) representation (L must be integer), which is given by $\hat{U} = \exp(-i \pi (\hat{L}_z + \hat{M}_z))$.
        It is a unitary operator that commutes with the Hamiltonian and has eigenvalues + 1 and - 1.
        Its action on the basis states of the spin-(L, L) representation, |m1, m2>, is given by $\hat{U}_1\ket{m_1,m_2}=(- 1)^{2 \left(m_1+m_2 \right)}\ket{m_1,m_2}$, where -L <= m1, m2 <= L.

        Returns
        -------
        U: numpy.ndarray
            The parity operator of the FP model in the spin-(L, L) representation, which is a (2L + 1)^2 x (2L + 1)^2 matrix.

        Example
        -------
        >>> FP_system = FP(0.5, 1)
        >>> FP_system.parity_operator()
        array([[ 1,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0, -1,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  1,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0, -1,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  1,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0, -1,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  1,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0, -1,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  1]])
        """
        D = int(2 * self._L + 1)
        M_list = np.arange(self._L, - self._L - 1, - 1)

        k = 0
        diag = np.zeros(D ** 2, dtype = int)
        for m1 in M_list:
            for m2 in M_list:
                diag[k] = (-1) ** int(m1 + m2)
                k += 1

        U = np.diag(diag)

        return U

    def exchange_operator(self) -> np.ndarray:
        r"""
        This function constructs the exchange operator of the FP model in the spin-(L, L) representation (L must be integer).
        It is a unitary operator that commutes with the Hamiltonian and has eigenvalues + 1 and - 1.
        Its action on the basis states of the spin-(L, L) representation, $\hat{U}_2\ket{m_1,m_2}=\ket{m_2,m_1}$, where -L <= m1, m2 <= L.

        Returns
        -------
        U: numpy.ndarray
            The exchange operator of the FP model in the spin-(L, L) representation, which is a (2L + 1)^2 x (2L + 1)^2 matrix.

        Example
        -------     
        >>> FP_system = FP(0.5, 1)
        >>> FP_system.exchange_operator()
        array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        """
        D = int(2 * self._L + 1)

        U = np.zeros((D ** 2, D ** 2), dtype = int)
        for i in range(D):
            for j in range(D):
                bra_index = i * D + j  
                ket_index = j * D + i 
                U[ket_index, bra_index] = 1
                
        return U

    def symmetry_blocks(self, spectrum: bool = False):
        """
        This function constructs the four parity blocks of the FP Hamiltonian in the spin-(L, L) representation (L must be integer), by using the parity and exchange operators to block-diagonalize the Hamiltonian.

        Parameters
        ----------
        spectrum: bool (optional)
            If True, the function also returns the spectra of the two parity blocks. The default value is False.

        Returns
        -------
        H_p_p: numpy.ndarray
            The (+, +) symmetry block of the FP Hamiltonian.  

        H_p_m: numpy.ndarray
            The (+, -) symmetry block of the FP Hamiltonian.
        
        H_m_p: numpy.ndarray
            The (-, +) symmetry block of the FP Hamiltonian.

        H_m_m: numpy.ndarray
            The (-, -) symmetry block of the FP Hamiltonian.

        E_p_p: numpy.ndarray (only if spectrum is True)
            The spectrum of the (+, +) symmetry block of the FP Hamiltonian.

        E_p_m: numpy.ndarray (only if spectrum is True)
            The spectrum of the (+, -) symmetry block of the FP Hamiltonian.

        E_m_p: numpy.ndarray (only if spectrum is True)
            The spectrum of the (-, +) symmetry block of the FP Hamiltonian.

        E_m_m: numpy.ndarray (only if spectrum is True)
            The spectrum of the (-, -) symmetry block of the FP Hamiltonian.

        Example
        -------
        >>> FP_system = FP(0.5, 1)
        >>> FP_system.build_intensive()
        >>> FP_system.symmetry_blocks()
        (array([[-3.        +0.j,  1.        +0.j,  0.        +0.j,
                  0.        +0.j],
                [ 1.        +0.j,  0.        +0.j,  1.        +0.j,
                  1.41421356+0.j],
                [ 0.        +0.j,  1.        +0.j,  3.        +0.j,
                  0.        +0.j],
                [ 0.        +0.j,  1.41421356+0.j,  0.        +0.j,
                  0.        +0.j]]),
        array([[0.+0.j]]),
        array([[-2.5+0.j, -1. +0.j],
               [-1. +0.j,  0.5+0.j]]),
        array([[-0.5+0.j, -1. +0.j],
               [-1. +0.j,  2.5+0.j]]))
        """
        if self._matrix is None:
            raise ValueError("Hamiltonian matrix has not been set.")

        U1 = self.parity_operator()
        U2 = self.exchange_operator()

        E_U2, phi_U2 = la.eigh(U2)

        idx_U2_m = E_U2 < 0
        idx_U2_p = E_U2 > 0

        H_FP_U2 = phi_U2.conj().T @ self.matrix @ phi_U2
        U1_U2 = phi_U2.conj().T @ U1 @ phi_U2

        H_FP_U2_m = H_FP_U2[idx_U2_m][:, idx_U2_m]
        H_FP_U2_p = H_FP_U2[idx_U2_p][:, idx_U2_p]

        U1_U2_m = U1_U2[idx_U2_m][:, idx_U2_m]
        U1_U2_p = U1_U2[idx_U2_p][:, idx_U2_p]

        E_U1_U2_m, phi_U1_U2_m = la.eigh(U1_U2_m)
        E_U1_U2_p, phi_U1_U2_p = la.eigh(U1_U2_p)

        idx_U1_m_U2_m = E_U1_U2_m < 0
        idx_U1_p_U2_m = E_U1_U2_m > 0
        idx_U1_m_U2_p = E_U1_U2_p < 0
        idx_U1_p_U2_p = E_U1_U2_p > 0

        H_FP_U1_U2_m = phi_U1_U2_m.conj().T @ H_FP_U2_m @ phi_U1_U2_m
        H_FP_U1_U2_p = phi_U1_U2_p.conj().T @ H_FP_U2_p @ phi_U1_U2_p

        H_FP_m_m = H_FP_U1_U2_m[idx_U1_m_U2_m][:, idx_U1_m_U2_m]
        H_FP_m_p = H_FP_U1_U2_p[idx_U1_m_U2_p][:, idx_U1_m_U2_p]
        H_FP_p_m = H_FP_U1_U2_m[idx_U1_p_U2_m][:, idx_U1_p_U2_m]
        H_FP_p_p = H_FP_U1_U2_p[idx_U1_p_U2_p][:, idx_U1_p_U2_p]

        dims_blocks = self.blocks_dimension()
        if H_FP_p_p.shape != (dims_blocks[0], dims_blocks[0]):
            raise ValueError("Dimension of the (+, +) symmetry block is wrong. Check the implementation of the parity and exchange operators and their spectra.")
        if H_FP_p_m.shape != (dims_blocks[1], dims_blocks[1]):
            raise ValueError("Dimension of the (+, -) symmetry block is wrong. Check the implementation of the parity and exchange operators and their spectra.")
        if H_FP_m_p.shape != (dims_blocks[2], dims_blocks[2]):
            raise ValueError("Dimension of the (-, +) symmetry block is wrong. Check the implementation of the parity and exchange operators and their spectra.")
        if H_FP_m_m.shape != (dims_blocks[3], dims_blocks[3]):
            raise ValueError("Dimension of the (-, -) symmetry block is wrong. Check the implementation of the parity and exchange operators and their spectra.")

        if spectrum:
            E_m_m, _ = la.eigh(H_FP_m_m)
            E_m_p, _ = la.eigh(H_FP_m_p)
            E_p_m, _ = la.eigh(H_FP_p_m)
            E_p_p, _ = la.eigh(H_FP_p_p)

            return H_FP_p_p, H_FP_p_m, H_FP_m_p, H_FP_m_m, E_p_p, E_p_m, E_m_p, E_m_m   

        return H_FP_p_p, H_FP_p_m, H_FP_m_p, H_FP_m_m

    def mean_level_spacing_ratio(self) -> tuple[float, float]:
        r"""
        This function returns the mean level-spacing ratio of the FP Hamiltonian in the (-, +) and (-, -) symmetry blocks, which are the only two blocks that exhibit Poisson to GOE level-spacing statistics transition as a function of the parameter \lambda.

        Returns
        -------
        mean_r_mp: float
            The mean level-spacing ratio of the (-, +) symmetry block of the FP Hamiltonian

        mean_r_mm: float
            The mean level-spacing ratio of the (-, -) symmetry block of the FP Hamiltonian
            
        Example
        -------
        >>> print(Jx_Jy_Jz(1/2))
        (array([[0. , 0.5],[0.5, 0. ]]), array([[0.+0.j , 0.-0.5j],[0.+0.5j, 0.+0.j ]]), array([[ 0.5,  0. ],[ 0. , -0.5]])) # Pauli matrices divided by 2
        """
        spectrum_blocks = self.symmetry_blocks(spectrum = True)

        spectrum_m_p = spectrum_blocks[6]
        spectrum_m_m = spectrum_blocks[7]

        diff_mp = spectrum_m_p[1 :] - spectrum_m_p[: - 1]
        diff_mm = spectrum_m_m[1 :] - spectrum_m_m[: - 1]

        r_array_mp = np.minimum(diff_mp[ : - 1], diff_mp[1 : ]) / np.maximum(diff_mp[ : - 1], diff_mp[1 :])
        r_array_mm = np.minimum(diff_mm[ : - 1], diff_mm[1 : ]) / np.maximum(diff_mm[ : - 1], diff_mm[1 :])
        
        mean_r_mp = np.mean(r_array_mp)
        mean_r_mm = np.mean(r_array_mm)

        return mean_r_mp, mean_r_mm

    def __str__(self) -> str:
        str_1 = f"FP model with a = {self._a}, L = {self._L}."
        str_2 = f"Hamiltonian is {self._model_type}." if self._model_type is not None else "Hamiltonian has not been built yet."

        return str_1 + " " + str_2 