from .Hamiltonian import Hamiltonian
from .util import spin_operators

import numpy as np
import scipy.linalg as la


class LMG(Hamiltonian):

    def __init__(self, h: float, J: float, S: float):
        """
        Constructor for the LMG class, which inherits from the Hamiltonian class.

        Parameters
        ----------
        h: float
            The transverse magnetic field  
        
        J: float
            The interaction strength    

        S: float (S = 1/2, 1, 3/2, 2, 5/2, ...)
            The angular momentum quantum number (which in the maximal spin sector is related to the system size N by S = N / 2)
        """
        super().__init__()
        self._h = h
        self._J = J
        self._S = S

        if S <= 0 or not np.isclose(2 * S, round(2 * S)):
            raise ValueError("S must be a positive integer or half-integer.")

    @property
    def h(self) -> float:
        return self._h

    @property
    def J(self) -> float:
        return self._J  

    @property
    def S(self) -> float:
        return self._S
    
    @property
    def matrix(self) -> np.ndarray | None:
        if self._matrix is None:
            raise ValueError("Hamiltonian matrix has not been set.")
        return self._matrix

    def build_extensive(self, N: int) -> np.ndarray:
        """
        The function constructs the extensive Lipkin-Meshkov-Glick (LMG) model Hamiltonian, see Eq. (2.4) in the main text.

        Parameters
        ----------
        N: int
            The system size

        Returns
        -------
        H: numpy.ndarray
            The extensive Hamiltonian of the LMG model in the spin-S representation, which is a (2S + 1) x (2S + 1) matrix
            
        Example
        -------
        >>> LMG_system = LMG(1.0, 1.0, 2)
        >>> LMG_system.build_extensive(4)
        array([[-2.        +0.j, -2.        +0.j,  0.        +0.j,
                 0.        +0.j,  0.        +0.j],
               [-2.        +0.j, -0.5       +0.j, -2.44948974+0.j,
                 0.        +0.j,  0.        +0.j],
               [ 0.        +0.j, -2.44948974+0.j,  0.        +0.j,
                -2.44948974+0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j, -2.44948974+0.j,
                -0.5       +0.j, -2.        +0.j],
               [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                -2.        +0.j, -2.        +0.j]])
        """
        Sx, _, Sz = spin_operators(self._S)
        
        if N <= 0:
            raise ValueError("N must be a positive integer for the extensive LMG model.")
        H = - (2 * self._J / N) * Sz @ Sz - 2 * self._h * Sx

        self._matrix = H
        self._model_type = "extensive"
        
        return H

    def build_intensive(self) -> np.ndarray:
        """
        The function constructs the intensive LMG model Hamiltonian, see Eq. (2.2) in the main text.

        Returns
        -------
        H: numpy.ndarray
            The intensive Hamiltonian of the LMG model in the spin-S representation, which is a (2S + 1) x (2S + 1) matrix
            
        Example
        -------
        >>> LMG_system = LMG(1.0, 1.0, 2)
        >>> LMG_system.build_intensive()
        array([[-0.5       +0.j, -0.5       +0.j,  0.        +0.j,
                 0.        +0.j,  0.        +0.j],
               [-0.5       +0.j, -0.125     +0.j, -0.61237244+0.j,
                 0.        +0.j,  0.        +0.j],
               [ 0.        +0.j, -0.61237244+0.j,  0.        +0.j,
                -0.61237244+0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j, -0.61237244+0.j,
                -0.125     +0.j, -0.5       +0.j],
               [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
                -0.5       +0.j, -0.5       +0.j]])
        """
        Sx, _, Sz = spin_operators(self._S)

        H = - self._J / (2 * self._S ** 2) * Sz @ Sz - self._h * Sx / self._S 

        self._matrix = H
        self._model_type = "intensive"
        
        return H

    def magnetization(self, level: int = 0, operator: np.ndarray | None = None) -> complex:
        """
        This function computes the magnetization of the eigenstate corresponding to the specified energy level, with respect to the specified operator. 
        If no operator is provided, it defaults to the rescaled z-component of the collective spin operator which is the order parameter of the LMG model.

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
            _, _, Sz = spin_operators(self._S)
            operator = Sz / self._S

        return self.expectation_value(level, operator)

    def blocks_dimension(self) -> tuple[int, int]:
        """
        This function computes the dimensions of the two parity blocks of the LMG Hamiltonian in the spin-S representation, which are labeled by the eigenvalues of the parity operator.
        The dimensions of the blocks are given by Eq. (B.12) in the main text.

        Returns
        -------
        dim_p: int
            Dimension of the (+) parity block of the LMG Hamiltonian.

        dim_m: int
            Dimension of the (-) parity block of the LMG Hamiltonian.

        Example
        -------
        >>> LMG_system = LMG(1.0, 1.0, 2)
        >>> LMG_system.blocks_dimension()
        (3, 2)
        """
        if self._S % 2 == 0:
            dim_p = self._S + 1
            dim_m = self._S
        
        elif self._S % 2 == 1:
            dim_p = self._S
            dim_m = self._S + 1

        else:
            dim_p = int(self._S + 0.5)
            dim_m = int(self._S + 0.5)

        return dim_p, dim_m

    def parity_operator(self) -> np.ndarray:
        r"""
        This function constructs the parity operator of the LMG model in the spin-S representation, which is given by $\hat{U} = \exp(-i \pi \hat{S}_x)$. 
        It is a unitary operator that commutes with the Hamiltonian and has eigenvalues + 1 and - 1.
        Its action on the basis states of the spin-S representation, |m>, is given by $\hat{U} |m> = \exp(-i \pi S) |-m>$, where -S <= m <= S.

        Returns
        -------
        U: numpy.ndarray
            The parity operator of the LMG model in the spin-S representation, which is a (2S + 1) x (2S + 1) matrix.

        Example
        -------
        >>> LMG_system = LMG(1.0, 1.0, 1.5)
        >>> LMG_system.parity_operator()
        array([[ 0.0000000e+00+0.j,  0.0000000e+00+0.j,  0.0000000e+00+0.j,
                -1.8369702e-16+1.j],
               [ 0.0000000e+00+0.j,  0.0000000e+00+0.j, -1.8369702e-16+1.j,
                 0.0000000e+00+0.j],
               [ 0.0000000e+00+0.j, -1.8369702e-16+1.j,  0.0000000e+00+0.j,
                 0.0000000e+00+0.j],
               [-1.8369702e-16+1.j,  0.0000000e+00+0.j,  0.0000000e+00+0.j,
                 0.0000000e+00+0.j]])
        """
        D = int(2 * self._S + 1)
        U = np.zeros((D, D), dtype = np.complex128)

        idx_ket = np.arange(D)
        idx_bra = D - 1 - idx_ket
        
        phase = np.exp(- 1j * np.pi * self._S)
        U[idx_bra,idx_ket] = phase
        
        return U

    def symmetry_blocks(self, spectrum: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function constructs the two parity blocks of the LMG Hamiltonian by using the parity operator to block-diagonalize the Hamiltonian.

        Parameters
        ----------
        spectrum: bool (optional)
            If True, the function also returns the spectra of the two parity blocks. The default value is False.

        Returns
        -------
        H_m: numpy.ndarray
            The (+) parity block of the LMG Hamiltonian.

        H_p: numpy.ndarray
            The (-) parity block of the LMG Hamiltonian.

        E_m: numpy.ndarray (only if spectrum is True)
            The spectrum of the (+) parity block of the LMG Hamiltonian.

        E_p: numpy.ndarray (only if spectrum is True)
            The spectrum of the (-) parity block of the LMG Hamiltonian.

        Example
        -------
        >>> LMG_system = LMG(1.0, 1.0, 1.5)
        >>> LMG_system.build_intensive()
        >>> LMG_system.symmetry_blocks()
        (array([[-5.00000000e-01+8.22571099e-35j, -1.06057524e-16-5.77350269e-01j],
                [-1.06057524e-16+5.77350269e-01j, -5.55555556e-02-4.51948026e-18j]]),
         array([[-5.00000000e-01-1.43281691e-33j,  1.06057524e-16+5.77350269e-01j],
                [ 1.06057524e-16-5.77350269e-01j, -5.55555556e-02-8.56462356e-18j]]))
        """
        if self._matrix is None:
            raise ValueError("Hamiltonian matrix has not been set.")

        U = self.parity_operator()
        E_U, phi_U = la.eigh(U)

        idx_m = E_U < 0
        idx_p = E_U > 0

        H_U = phi_U.conj().T @ self.matrix @ phi_U
        H_m = H_U[idx_m][:, idx_m]
        H_p  = H_U[idx_p][:, idx_p]

        dims_blocks = self.blocks_dimension()
        if H_p.shape != (dims_blocks[0], dims_blocks[0]):
            raise ValueError("Dimension of the (+) parity block is wrong. Check the implementation of the parity operator and its spectrum.")
        if H_m.shape != (dims_blocks[1], dims_blocks[1]):
            raise ValueError("Dimension of the (-) parity block is wrong. Check the implementation of the parity operator and its spectrum.")

        if spectrum:
            E_m, _ = la.eigh(H_m)
            E_p, _ = la.eigh(H_p)
            return H_m, H_p, E_m, E_p, 

        return H_m, H_p

    def level_spacing_ratio(self) -> tuple[float, float]:
        r"""
        This function returns the mean level-spacing ratio of the LMG Hamiltonian of the full spectrum and of the two parity blocks.

        Returns
        -------
        r_array: numpy.ndarray
            The level-spacing ratio of the full spectrum of the LMG Hamiltonian.

        r_array_p: numpy.ndarray
            The level-spacing ratio of the spectrum of the (+) parity block of the LMG Hamiltonian.

        r_array_m: numpy.ndarray
            The level-spacing ratio of the spectrum of the (-) parity block of the LMG Hamiltonian.
            
        Example
        -------
        >>> print(Jx_Jy_Jz(1/2))
        (array([[0. , 0.5],[0.5, 0. ]]), array([[0.+0.j , 0.-0.5j],[0.+0.5j, 0.+0.j ]]), array([[ 0.5,  0. ],[ 0. , -0.5]])) # Pauli matrices divided by 2
        """
        spectrum_blocks = self.symmetry_blocks(spectrum = True)
        spectrum_p = spectrum_blocks[2]
        spectrum_m = spectrum_blocks[3]
        E_spectrum, _ = la.eigh(self.matrix)

        diff_p = spectrum_p[1 :] - spectrum_p[: - 1]
        diff_m = spectrum_m[1 :] - spectrum_m[: - 1]
        diff = E_spectrum[1 :] - E_spectrum[: - 1]

        r_array_p = np.minimum(diff_p[ : - 1], diff_p[1 : ]) / np.maximum(diff_p[ : - 1], diff_p[1 :])
        r_array_m = np.minimum(diff_m[ : - 1], diff_m[1 : ]) / np.maximum(diff_m[ : - 1], diff_m[1 :])
        r_array = np.minimum(diff[ : - 1], diff[1 : ]) / np.maximum(diff[ : - 1], diff[1 :])

        return r_array, r_array_p, r_array_m

    def __str__(self) -> str:
        str_1 = f"LMG model with h = {self._h}, J = {self._J}, S = {self._S}."
        str_2 = f"Hamiltonian is {self._model_type}." if self._model_type is not None else "Hamiltonian has not been built yet."
        
        return str_1 + " " + str_2 