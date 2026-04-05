"""
engine/poisson.py
=================
Spectral Poisson solver for gravitational self-interaction.
Solves nabla^2 Phi = 4*pi*G * rho in a periodic box using FFT.
"""

import numpy as np
from numpy.fft import fftn, ifftn


def solve_poisson(rho, K2, G):
    """
    Solve the Poisson equation for gravitational potential.

    Parameters
    ----------
    rho : ndarray (N, N, N)
        Energy density field (source term).
    K2 : ndarray (N, N, N)
        Precomputed k^2 grid from SexticEvolver.
    G : float
        Gravitational coupling constant.

    Returns
    -------
    Phi : ndarray (N, N, N)
        Gravitational potential.
    """
    rho_hat = fftn(rho)
    # Avoid division by zero at k=0 (DC mode)
    K2_safe = K2.copy()
    K2_safe[0, 0, 0] = 1.0  # placeholder to avoid div-by-zero
    Phi_hat = -4.0 * np.pi * G * rho_hat / K2_safe
    Phi_hat[0, 0, 0] = 0.0  # zero mean potential
    return np.real(ifftn(Phi_hat))
