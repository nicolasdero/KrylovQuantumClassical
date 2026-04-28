import numpy as np


def spin_operators(S: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function returns the x, y and z-components of the angular momentum operator in the spin-S representation.

    Parameters
    ----------
    S: float (S = 1/2, 1, 3/2, 2, 5/2, ...)
        The angular momentum quantum number

    Returns
    -------
    Sx: numpy.ndarray
        x-component of the angular momentum operator in the spin-S representation

    Sy: numpy.ndarray
        y-component of the angular momentum operator in the spin-S representation

    Sz: numpy.ndarray
        z-component of the angular momentum operator in the spin-S representation
        
    Example
    -------
    >>> spin_operators(0.5)
    (array([[0. +0.j, 0.5+0.j],
            [0.5+0.j, 0. +0.j]]),
     array([[0.+0.j , 0.-0.5j],
            [0.+0.5j, 0.+0.j ]]),
     array([[ 0.5,  0. ],
            [ 0. , -0.5]])) # Pauli matrices divided by 2
    """
    if S < 0 or not np.isclose(2 * S, round(2 * S)):
        raise ValueError("S (or L) must be a non-negative integer or half-integer.")

    D = int(2 * S + 1)
    S_p = np.zeros((D, D), dtype = np.complex128)
    S_m = np.zeros((D, D), dtype = np.complex128)

    M_list = np.arange(S, - S - 1, - 1)
    M_p_list = np.delete(M_list - 1, - 1)
    M_m_list = np.delete(M_list + 1, 0)

    i_index = np.arange(len(M_p_list))

    S_p[i_index, i_index + 1] = np.sqrt(S * (S + 1) - M_p_list * (M_p_list + 1)) # This is S_+
    S_m[i_index + 1, i_index] = np.sqrt(S * (S + 1) - M_m_list * (M_m_list - 1)) # This is S_-

    Sx = (S_p + S_m) / 2
    Sy = (S_p - S_m) / (2j)
    Sz = np.diag(M_list)

    return Sx, Sy, Sz