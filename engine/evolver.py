"""
engine/evolver.py
=================
Standalone reusable module containing SexticEvolver: a 3D spectral RK4 time-evolution
engine for a scalar field with sextic potential V(phi) = 1/2*m^2*phi^2 - (g4/24)*phi^4
+ (g6/720)*phi^6, with Kreiss-Oliger dissipation.

Includes an evolve() loop with incremental checkpointing and resume support.
"""

import base64
import sys
import time

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq


# ---------------------------------------------------------------------------
#  Field serialization helpers (for checkpoint JSON)
# ---------------------------------------------------------------------------

def serialize_field(arr):
    """Encode a numpy float64 array as a base64 string for JSON storage."""
    return base64.b64encode(arr.tobytes()).decode('ascii')


def deserialize_field(s, shape):
    """Decode a base64 string back into a numpy float64 array."""
    return np.frombuffer(base64.b64decode(s), dtype=np.float64).reshape(shape).copy()


class SexticEvolver:
    """
    Evolves the field using RK4 with a sextic potential and Kreiss-Oliger dissipation.

    Potential: V(phi) = 1/2*m^2*phi^2 - (g4/24)*phi^4 + (g6/720)*phi^6
    EOM: phi_tt = nabla^2 phi - V'(phi) - dissipation
    """
    def __init__(self, N, L, m, g4, g6, dissipation_sigma=0.01):
        self.N = N
        self.L = L
        self.dx = L / N
        self.m_sq = m**2
        self.g4 = g4
        self.g6 = g6
        self.sigma = dissipation_sigma

        # Periodic grid
        coords = np.linspace(-L/2, L/2, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(coords, coords, coords, indexing='ij')

        # Precompute spectral operators
        k = fftfreq(N, d=self.dx) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K2 = KX**2 + KY**2 + KZ**2
        self.K4 = self.K2**2

        self.phi = np.zeros((N, N, N))
        self.phi_dot = np.zeros((N, N, N))
        self.t = 0.0

    def set_initial_conditions(self, phi, phi_dot):
        """Set initial field and its time derivative."""
        self.phi = phi.copy()
        self.phi_dot = phi_dot.copy()
        self.t = 0.0

    def compute_rhs(self, phi, phi_dot):
        """Compute the RHS of the coupled first-order system."""
        phi_hat = fftn(phi)
        laplacian = np.real(ifftn(-self.K2 * phi_hat))

        # V'(phi) = m^2*phi - (g4/6)*phi^3 + (g6/120)*phi^5
        v_prime = self.m_sq * phi - (self.g4 / 6.0) * phi**3 + (self.g6 / 120.0) * phi**5

        # Kreiss-Oliger dissipation: -sigma * dx^4 * laplacian(laplacian(phi))
        dissipation = self.sigma * self.dx**4 * np.real(ifftn(self.K4 * phi_hat))

        phi_ddot = laplacian - v_prime - dissipation
        return phi_dot, phi_ddot

    def step_rk4(self, dt):
        """Advance one time step using RK4."""
        phi0 = self.phi.copy()
        dot0 = self.phi_dot.copy()

        p1, v1 = self.compute_rhs(phi0, dot0)
        p2, v2 = self.compute_rhs(phi0 + 0.5 * dt * p1, dot0 + 0.5 * dt * v1)
        p3, v3 = self.compute_rhs(phi0 + 0.5 * dt * p2, dot0 + 0.5 * dt * v2)
        p4, v4 = self.compute_rhs(phi0 + dt * p3, dot0 + dt * v3)

        self.phi = phi0 + (dt / 6.0) * (p1 + 2*p2 + 2*p3 + p4)
        self.phi_dot = dot0 + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
        self.t += dt

    def compute_energy(self):
        """Compute total energy (Hamiltonian)."""
        dV = self.dx**3
        # Kinetic: 1/2 * integral(phi_dot^2)
        E_kin = 0.5 * np.sum(self.phi_dot**2) * dV

        # Gradient: 1/2 * integral(|grad phi|^2) - computed in Fourier space
        phi_hat = fftn(self.phi)
        E_grad = 0.5 * np.sum(self.K2 * np.abs(phi_hat)**2) * dV / (self.N**3)

        # Potential: integral(V(phi))
        v = 0.5 * self.m_sq * self.phi**2 - (self.g4 / 24.0) * self.phi**4 + (self.g6 / 720.0) * self.phi**6
        E_pot = np.sum(v) * dV

        return float(E_kin + E_grad + E_pot)

    # ------------------------------------------------------------------
    #  High-level evolution loop with checkpointing
    # ------------------------------------------------------------------

    def evolve(self, dt, n_steps, record_every=10, checkpoint_every=1000,
               checkpoint_callback=None, resume_from=None, print_every=1000,
               tag="", extra_diagnostic_fn=None):
        """Run the evolution loop with optional checkpointing and resume.

        Args:
            dt: time step
            n_steps: total number of RK4 steps
            record_every: record diagnostics every N steps
            checkpoint_every: call checkpoint_callback every N steps
            checkpoint_callback: callable(state_dict) for incremental saves
            resume_from: checkpoint state dict to resume from
            print_every: print progress every N steps
            tag: label for progress messages (empty string disables printing)
            extra_diagnostic_fn: callable(evolver) -> dict, called at each
                recording point; results stored in state['extra_diagnostics']

        Returns:
            State dict with time_series and metadata.
            Does NOT include phi_b64/phi_dot_b64 (those are checkpoint-only).
        """
        if resume_from is not None:
            start_step = resume_from['completed_steps'] + 1
            self.phi = deserialize_field(
                resume_from['phi_b64'], self.phi.shape)
            self.phi_dot = deserialize_field(
                resume_from['phi_dot_b64'], self.phi.shape)
            self.t = resume_from['t']
            times = list(resume_from['time_series']['times'])
            E_total = list(resume_from['time_series']['E_total'])
            max_amplitude = list(resume_from['time_series']['max_amplitude'])
            extra_diagnostics = list(
                resume_from.get('extra_diagnostics') or [])
            E0 = resume_from['E0']
            max_phi0 = resume_from['max_phi0']
            wall_prior = resume_from.get('wall_elapsed', 0.0)
            if tag:
                print("  [%s] RESUMING from step %d (t=%.1f)" % (
                    tag, start_step - 1, self.t))
                sys.stdout.flush()
        else:
            start_step = 0
            times = []
            E_total = []
            max_amplitude = []
            extra_diagnostics = []
            E0 = self.compute_energy()
            max_phi0 = float(np.max(np.abs(self.phi)))
            wall_prior = 0.0

        wall_start = time.perf_counter()

        for step in range(start_step, n_steps + 1):
            # --- record diagnostics ---
            if step % record_every == 0:
                E_now = self.compute_energy()
                amp = float(np.max(np.abs(self.phi)))
                times.append(float(self.t))
                E_total.append(E_now)
                max_amplitude.append(amp)
                if extra_diagnostic_fn is not None:
                    extra_diagnostics.append(extra_diagnostic_fn(self))

            # --- progress print ---
            if tag and step % print_every == 0:
                wall_now = time.perf_counter() - wall_start + wall_prior
                steps_done = step - start_step
                if steps_done > 0 and step < n_steps:
                    rate = (time.perf_counter() - wall_start) / steps_done
                    eta = rate * (n_steps - step)
                    drift = abs(E_total[-1] - E0) / (abs(E0) + 1e-30)
                    print("  [%s] step=%5d/%d  t=%7.1f  E=%.5e"
                          "  drift=%.2e  elapsed=%.0fs  ETA=%.0fs" % (
                              tag, step, n_steps, self.t, E_total[-1],
                              drift, wall_now, eta))
                elif E_total:
                    print("  [%s] step=%5d/%d  t=%7.1f  E=%.5e" % (
                        tag, step, n_steps, self.t, E_total[-1]))
                sys.stdout.flush()

            # --- advance field ---
            if step < n_steps:
                self.step_rk4(dt)

            # --- checkpoint (after step_rk4 so field is at next step) ---
            if (checkpoint_callback is not None
                    and step % checkpoint_every == 0
                    and step > 0
                    and step < n_steps):
                wall_now = time.perf_counter() - wall_start + wall_prior
                ckpt = {
                    'completed': False,
                    'completed_steps': step,
                    't': float(self.t),
                    'E0': E0,
                    'max_phi0': max_phi0,
                    'wall_elapsed': wall_now,
                    'phi_b64': serialize_field(self.phi),
                    'phi_dot_b64': serialize_field(self.phi_dot),
                    'time_series': {
                        'times': list(times),
                        'E_total': list(E_total),
                        'max_amplitude': list(max_amplitude),
                    },
                }
                if extra_diagnostics:
                    ckpt['extra_diagnostics'] = list(extra_diagnostics)
                checkpoint_callback(ckpt)

        wall_time = time.perf_counter() - wall_start + wall_prior

        state = {
            'completed': True,
            'completed_steps': n_steps,
            't': float(self.t),
            'E0': E0,
            'max_phi0': max_phi0,
            'wall_elapsed': wall_time,
            'time_series': {
                'times': times,
                'E_total': E_total,
                'max_amplitude': max_amplitude,
            },
        }
        if extra_diagnostics:
            state['extra_diagnostics'] = extra_diagnostics
        return state
