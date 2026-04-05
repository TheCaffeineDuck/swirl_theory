"""
engine/formation_evolver.py
============================
Extended evolver with gravitational self-interaction for formation studies.
Subclasses SexticEvolver, adding a spectral Poisson gravity term.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq

from .evolver import SexticEvolver
from .poisson import solve_poisson


class FormationEvolver(SexticEvolver):
    """
    SexticEvolver + weak gravitational self-interaction.

    The gravitational force term is: -grad(Phi) . grad(phi)
    where Phi solves nabla^2 Phi = 4*pi*G * rho, with rho = energy density.

    When G_coupling=0, this reduces exactly to SexticEvolver (gravity skipped).
    """

    def __init__(self, N, L, m, g4, g6, dissipation_sigma=0.01,
                 G_coupling=0.001):
        super().__init__(N, L, m, g4, g6, dissipation_sigma)
        self.G_coupling = G_coupling

        # Precompute per-axis k vectors for gradient computation
        k = fftfreq(N, d=self.dx) * 2 * np.pi
        self.kx = k.reshape(N, 1, 1)
        self.ky = k.reshape(1, N, 1)
        self.kz = k.reshape(1, 1, N)

    def _compute_energy_density(self, phi, phi_dot):
        """Compute per-cell energy density rho(x,y,z)."""
        phi_hat = fftn(phi)

        # Kinetic
        rho = 0.5 * phi_dot ** 2

        # Gradient (spectral per-axis)
        grad_x = np.real(ifftn(1j * self.kx * phi_hat))
        grad_y = np.real(ifftn(1j * self.ky * phi_hat))
        grad_z = np.real(ifftn(1j * self.kz * phi_hat))
        rho += 0.5 * (grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

        # Potential
        rho += (0.5 * self.m_sq * phi ** 2
                - (self.g4 / 24.0) * phi ** 4
                + (self.g6 / 720.0) * phi ** 6)

        return rho

    def compute_rhs(self, phi, phi_dot):
        """Compute RHS with optional gravitational force term."""
        # Base field dynamics (unchanged)
        dphi, dphi_dot = super().compute_rhs(phi, phi_dot)

        # Skip gravity if coupling is zero
        if self.G_coupling == 0.0:
            return dphi, dphi_dot

        # Gravitational self-interaction
        rho = self._compute_energy_density(phi, phi_dot)
        Phi = solve_poisson(rho, self.K2, self.G_coupling)

        # Spectral gradients of Phi and phi
        Phi_hat = fftn(Phi)
        phi_hat = fftn(phi)

        grad_Phi_x = np.real(ifftn(1j * self.kx * Phi_hat))
        grad_Phi_y = np.real(ifftn(1j * self.ky * Phi_hat))
        grad_Phi_z = np.real(ifftn(1j * self.kz * Phi_hat))

        grad_phi_x = np.real(ifftn(1j * self.kx * phi_hat))
        grad_phi_y = np.real(ifftn(1j * self.ky * phi_hat))
        grad_phi_z = np.real(ifftn(1j * self.kz * phi_hat))

        # Force: -grad(Phi) . grad(phi)
        grav_force = -(grad_Phi_x * grad_phi_x
                       + grad_Phi_y * grad_phi_y
                       + grad_Phi_z * grad_phi_z)

        dphi_dot = dphi_dot + grav_force

        return dphi, dphi_dot
