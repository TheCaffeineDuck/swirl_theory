"""
engine/complex_evolver.py
=========================
Complex scalar field evolver for Q-ball simulations.

Evolves two real fields (phi_R, phi_I) representing a complex scalar
phi = phi_R + i*phi_I with U(1) symmetric sextic potential:

  V(|phi|^2) = m^2 |phi|^2 / 2 - g4 |phi|^4 / 24 + g6 |phi|^6 / 720

where |phi|^2 = phi_R^2 + phi_I^2.

EOM:
  d^2_t phi_R = nabla^2 phi_R - dV/dphi_R - dissipation
  d^2_t phi_I = nabla^2 phi_I - dV/dphi_I - dissipation

where dV/dphi_R = [m^2 - (g4/6)|phi|^2 + (g6/120)|phi|^4] phi_R
(and similarly for phi_I).

Conserved Noether charge: Q = integral(phi_R * d_t phi_I - phi_I * d_t phi_R) d^3x

Uses spectral Laplacian and RK4 time integration, same as SexticEvolver.
"""

import base64
import sys
import time

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq


# ---------------------------------------------------------------------------
#  Field serialization helpers
# ---------------------------------------------------------------------------

def serialize_field(arr):
    """Encode a numpy float64 array as a base64 string for JSON storage."""
    return base64.b64encode(arr.tobytes()).decode('ascii')


def deserialize_field(s, shape):
    """Decode a base64 string back into a numpy float64 array."""
    return np.frombuffer(base64.b64decode(s), dtype=np.float64).reshape(shape).copy()


class ComplexSexticEvolver:
    """
    Evolves a complex scalar field using RK4 with sextic potential
    and Kreiss-Oliger dissipation.

    Potential: V(|phi|^2) = m^2|phi|^2/2 - (g4/24)|phi|^4 + (g6/720)|phi|^6
    EOM: phi_R_tt = nabla^2 phi_R - [m^2 - (g4/6)|phi|^2 + (g6/120)|phi|^4] phi_R
         phi_I_tt = nabla^2 phi_I - [m^2 - (g4/6)|phi|^2 + (g6/120)|phi|^4] phi_I
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
        coords = np.linspace(-L / 2, L / 2, N, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(coords, coords, coords, indexing='ij')

        # Precompute spectral operators
        k = fftfreq(N, d=self.dx) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        self.K2 = KX**2 + KY**2 + KZ**2
        self.K4 = self.K2**2

        # Fields: real and imaginary parts
        self.phi_R = np.zeros((N, N, N))
        self.phi_I = np.zeros((N, N, N))
        self.pi_R = np.zeros((N, N, N))   # d_t phi_R
        self.pi_I = np.zeros((N, N, N))   # d_t phi_I
        self.t = 0.0

    def set_initial_conditions(self, phi_R, phi_I, pi_R, pi_I):
        """Set initial field components and their time derivatives."""
        self.phi_R = phi_R.copy()
        self.phi_I = phi_I.copy()
        self.pi_R = pi_R.copy()
        self.pi_I = pi_I.copy()
        self.t = 0.0

    def compute_rhs(self, phi_R, phi_I, pi_R, pi_I):
        """Compute the RHS of the coupled first-order system.

        Returns (d_t phi_R, d_t phi_I, d_t pi_R, d_t pi_I).
        """
        # Spectral Laplacians
        phi_R_hat = fftn(phi_R)
        phi_I_hat = fftn(phi_I)
        lap_R = np.real(ifftn(-self.K2 * phi_R_hat))
        lap_I = np.real(ifftn(-self.K2 * phi_I_hat))

        # |phi|^2 = phi_R^2 + phi_I^2
        mod_sq = phi_R**2 + phi_I**2
        mod_4 = mod_sq**2

        # Effective mass term: m^2 - (g4/6)|phi|^2 + (g6/120)|phi|^4
        mass_eff = self.m_sq - (self.g4 / 6.0) * mod_sq + (self.g6 / 120.0) * mod_4

        # dV/dphi_R = mass_eff * phi_R, dV/dphi_I = mass_eff * phi_I
        v_prime_R = mass_eff * phi_R
        v_prime_I = mass_eff * phi_I

        # Kreiss-Oliger dissipation: -sigma * dx^4 * nabla^4(phi)
        diss_R = self.sigma * self.dx**4 * np.real(ifftn(self.K4 * phi_R_hat))
        diss_I = self.sigma * self.dx**4 * np.real(ifftn(self.K4 * phi_I_hat))

        # Accelerations
        pi_R_dot = lap_R - v_prime_R - diss_R
        pi_I_dot = lap_I - v_prime_I - diss_I

        return pi_R, pi_I, pi_R_dot, pi_I_dot

    def step_rk4(self, dt):
        """Advance one time step using RK4."""
        pR0 = self.phi_R.copy()
        pI0 = self.phi_I.copy()
        vR0 = self.pi_R.copy()
        vI0 = self.pi_I.copy()

        # k1
        dpR1, dpI1, dvR1, dvI1 = self.compute_rhs(pR0, pI0, vR0, vI0)

        # k2
        dpR2, dpI2, dvR2, dvI2 = self.compute_rhs(
            pR0 + 0.5 * dt * dpR1, pI0 + 0.5 * dt * dpI1,
            vR0 + 0.5 * dt * dvR1, vI0 + 0.5 * dt * dvI1)

        # k3
        dpR3, dpI3, dvR3, dvI3 = self.compute_rhs(
            pR0 + 0.5 * dt * dpR2, pI0 + 0.5 * dt * dpI2,
            vR0 + 0.5 * dt * dvR2, vI0 + 0.5 * dt * dvI2)

        # k4
        dpR4, dpI4, dvR4, dvI4 = self.compute_rhs(
            pR0 + dt * dpR3, pI0 + dt * dpI3,
            vR0 + dt * dvR3, vI0 + dt * dvI3)

        # Update
        self.phi_R = pR0 + (dt / 6.0) * (dpR1 + 2*dpR2 + 2*dpR3 + dpR4)
        self.phi_I = pI0 + (dt / 6.0) * (dpI1 + 2*dpI2 + 2*dpI3 + dpI4)
        self.pi_R = vR0 + (dt / 6.0) * (dvR1 + 2*dvR2 + 2*dvR3 + dvR4)
        self.pi_I = vI0 + (dt / 6.0) * (dvI1 + 2*dvI2 + 2*dvI3 + dvI4)
        self.t += dt

    def compute_energy(self):
        """Compute total energy (Hamiltonian) of the complex field."""
        dV = self.dx**3

        # Kinetic: 1/2 * integral(pi_R^2 + pi_I^2)
        E_kin = 0.5 * np.sum(self.pi_R**2 + self.pi_I**2) * dV

        # Gradient: 1/2 * integral(|grad phi_R|^2 + |grad phi_I|^2)
        phi_R_hat = fftn(self.phi_R)
        phi_I_hat = fftn(self.phi_I)
        E_grad = 0.5 * (np.sum(self.K2 * np.abs(phi_R_hat)**2) +
                        np.sum(self.K2 * np.abs(phi_I_hat)**2)) * dV / (self.N**3)

        # Potential: V(|phi|^2) = m^2|phi|^2/2 - g4|phi|^4/24 + g6|phi|^6/720
        mod_sq = self.phi_R**2 + self.phi_I**2
        V = (0.5 * self.m_sq * mod_sq
             - (self.g4 / 24.0) * mod_sq**2
             + (self.g6 / 720.0) * mod_sq**3)
        E_pot = np.sum(V) * dV

        return float(E_kin + E_grad + E_pot)

    def compute_charge(self):
        """Compute the conserved Noether charge Q.

        Q = integral(phi_R * pi_I - phi_I * pi_R) d^3x
        """
        dV = self.dx**3
        integrand = self.phi_R * self.pi_I - self.phi_I * self.pi_R
        return float(np.sum(integrand) * dV)

    def compute_max_amplitude(self):
        """Compute max |phi| on the grid."""
        mod_sq = self.phi_R**2 + self.phi_I**2
        return float(np.sqrt(np.max(mod_sq)))

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
            tag: label for progress messages
            extra_diagnostic_fn: callable(evolver) -> dict

        Returns:
            State dict with time_series and metadata.
        """
        if resume_from is not None:
            start_step = resume_from['completed_steps'] + 1
            self.phi_R = deserialize_field(
                resume_from['phi_R_b64'], self.phi_R.shape)
            self.phi_I = deserialize_field(
                resume_from['phi_I_b64'], self.phi_I.shape)
            self.pi_R = deserialize_field(
                resume_from['pi_R_b64'], self.pi_R.shape)
            self.pi_I = deserialize_field(
                resume_from['pi_I_b64'], self.pi_I.shape)
            self.t = resume_from['t']
            times = list(resume_from['time_series']['times'])
            E_total = list(resume_from['time_series']['E_total'])
            Q_total = list(resume_from['time_series']['Q_total'])
            max_amplitude = list(resume_from['time_series']['max_amplitude'])
            extra_diagnostics = list(
                resume_from.get('extra_diagnostics') or [])
            E0 = resume_from['E0']
            Q0 = resume_from['Q0']
            wall_prior = resume_from.get('wall_elapsed', 0.0)
            if tag:
                print("  [%s] RESUMING from step %d (t=%.1f)" % (
                    tag, start_step - 1, self.t))
                sys.stdout.flush()
        else:
            start_step = 0
            times = []
            E_total = []
            Q_total = []
            max_amplitude = []
            extra_diagnostics = []
            E0 = self.compute_energy()
            Q0 = self.compute_charge()
            wall_prior = 0.0

        wall_start = time.perf_counter()

        for step in range(start_step, n_steps + 1):
            # --- record diagnostics ---
            if step % record_every == 0:
                E_now = self.compute_energy()
                Q_now = self.compute_charge()
                amp = self.compute_max_amplitude()
                times.append(float(self.t))
                E_total.append(E_now)
                Q_total.append(Q_now)
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
                    E_drift = abs(E_total[-1] - E0) / (abs(E0) + 1e-30)
                    Q_drift = abs(Q_total[-1] - Q0) / (abs(Q0) + 1e-30)
                    print("  [%s] step=%5d/%d  t=%7.1f  E=%.5e  dE=%.2e"
                          "  Q=%.5e  dQ=%.2e  elapsed=%.0fs  ETA=%.0fs" % (
                              tag, step, n_steps, self.t, E_total[-1],
                              E_drift, Q_total[-1], Q_drift, wall_now, eta))
                elif E_total:
                    print("  [%s] step=%5d/%d  t=%7.1f  E=%.5e  Q=%.5e" % (
                        tag, step, n_steps, self.t, E_total[-1], Q_total[-1]))
                sys.stdout.flush()

            # --- advance field ---
            if step < n_steps:
                self.step_rk4(dt)

            # --- checkpoint ---
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
                    'Q0': Q0,
                    'wall_elapsed': wall_now,
                    'phi_R_b64': serialize_field(self.phi_R),
                    'phi_I_b64': serialize_field(self.phi_I),
                    'pi_R_b64': serialize_field(self.pi_R),
                    'pi_I_b64': serialize_field(self.pi_I),
                    'time_series': {
                        'times': list(times),
                        'E_total': list(E_total),
                        'Q_total': list(Q_total),
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
            'Q0': Q0,
            'wall_elapsed': wall_time,
            'time_series': {
                'times': times,
                'E_total': E_total,
                'Q_total': Q_total,
                'max_amplitude': max_amplitude,
            },
        }
        if extra_diagnostics:
            state['extra_diagnostics'] = extra_diagnostics
        return state
