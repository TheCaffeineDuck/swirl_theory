"""
engine/configuration_detector.py
=================================
Oscillon tracker and graph analyzer for formation studies.
Detects oscillon centers, assigns phases, builds neighbor graph,
and computes cross-edge statistics.
"""

import numpy as np
from scipy.ndimage import maximum_filter


def _compute_energy_density(phi, phi_dot, evolver):
    """Compute per-cell energy density H(x,y,z)."""
    from numpy.fft import fftn, ifftn

    dV = evolver.dx ** 3
    phi_hat = fftn(phi)

    # Kinetic energy density
    H = 0.5 * phi_dot ** 2

    # Gradient energy density (computed in Fourier space, per-cell)
    for axis in range(3):
        k_ax = np.fft.fftfreq(evolver.N, d=evolver.dx) * 2 * np.pi
        shape = [1, 1, 1]
        shape[axis] = evolver.N
        k_component = k_ax.reshape(shape)
        grad_component = np.real(ifftn(1j * k_component * phi_hat))
        H += 0.5 * grad_component ** 2

    # Potential energy density
    H += (0.5 * evolver.m_sq * phi ** 2
          - (evolver.g4 / 24.0) * phi ** 4
          + (evolver.g6 / 720.0) * phi ** 6)

    return H


def _find_oscillon_centers(H, evolver, threshold_frac=0.1, d_min=3.0):
    """Find oscillon centers as local maxima of energy density."""
    if np.max(H) == 0:
        return np.array([]).reshape(0, 3)

    threshold = threshold_frac * np.max(H)

    # Local maximum detection using a filter window
    filter_size = max(3, int(d_min / evolver.dx))
    if filter_size % 2 == 0:
        filter_size += 1  # ensure odd
    local_max = maximum_filter(H, size=filter_size)
    peaks_mask = (H == local_max) & (H > threshold)

    peak_indices = np.argwhere(peaks_mask)  # (n_peaks, 3) array of grid indices
    if len(peak_indices) == 0:
        return np.array([]).reshape(0, 3)

    # Convert grid indices to physical coordinates
    coords = np.linspace(-evolver.L / 2, evolver.L / 2,
                         evolver.N, endpoint=False)
    centers = np.array([[coords[ix], coords[iy], coords[iz]]
                        for ix, iy, iz in peak_indices])

    # Cluster nearby peaks: keep the strongest in each cluster
    H_values = np.array([H[ix, iy, iz] for ix, iy, iz in peak_indices])
    order = np.argsort(-H_values)  # strongest first
    keep = []
    for idx in order:
        pos = centers[idx]
        too_close = False
        for kept_idx in keep:
            dist = np.linalg.norm(pos - centers[kept_idx])
            if dist < d_min:
                too_close = True
                break
        if not too_close:
            keep.append(idx)

    return centers[keep]


def detect_configuration(phi, phi_dot, evolver, t,
                         threshold_frac=0.1, d_min=3.0, d_neighbor=8.0):
    """
    Full diagnostic: detect oscillons, assign phases, build graph,
    compute cross-edge fraction.

    Parameters
    ----------
    phi : ndarray (N, N, N)
        Current field.
    phi_dot : ndarray (N, N, N)
        Current field time derivative.
    evolver : SexticEvolver or FormationEvolver
        Evolver instance (for grid info and energy computation).
    t : float
        Current simulation time.
    threshold_frac : float
        Fraction of max energy density used as detection threshold.
    d_min : float
        Minimum separation for clustering peaks.
    d_neighbor : float
        Maximum distance for two oscillons to be considered neighbors.

    Returns
    -------
    dict
        Diagnostic dictionary with keys: time, n_oscillons, centers, phases,
        n_edges, n_cross_edges, cross_edge_fraction, total_energy, max_amplitude.
    """
    H = _compute_energy_density(phi, phi_dot, evolver)
    centers = _find_oscillon_centers(H, evolver, threshold_frac, d_min)

    n_osc = len(centers)

    # Step B: determine phases from sign of phi at each center
    phases = []
    coords = np.linspace(-evolver.L / 2, evolver.L / 2,
                         evolver.N, endpoint=False)
    for cx, cy, cz in centers:
        # Find nearest grid index
        ix = int(np.argmin(np.abs(coords - cx)))
        iy = int(np.argmin(np.abs(coords - cy)))
        iz = int(np.argmin(np.abs(coords - cz)))
        phases.append(0 if phi[ix, iy, iz] >= 0 else 1)

    # Step C: build neighbor graph
    n_edges = 0
    n_cross_edges = 0
    if n_osc >= 2:
        for i in range(n_osc):
            for j in range(i + 1, n_osc):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < d_neighbor:
                    n_edges += 1
                    if phases[i] != phases[j]:
                        n_cross_edges += 1

    cross_edge_fraction = n_cross_edges / n_edges if n_edges > 0 else 0.0

    # Total energy
    dV = evolver.dx ** 3
    total_energy = float(np.sum(H) * dV)

    return {
        "time": float(t),
        "n_oscillons": n_osc,
        "centers": centers.tolist(),
        "phases": phases,
        "n_edges": n_edges,
        "n_cross_edges": n_cross_edges,
        "cross_edge_fraction": float(cross_edge_fraction),
        "total_energy": total_energy,
        "max_amplitude": float(np.max(np.abs(phi))),
    }
