"""
engine/random_initial_conditions.py
====================================
Generate random multi-oscillon initial conditions for formation studies.
"""

import numpy as np


def generate_random_oscillons(N_osc, N_grid, L, phi0, R,
                               min_separation=6.0, seed=None,
                               max_retries=1000):
    """
    Place N_osc oscillons at random positions with random phases.

    Parameters
    ----------
    N_osc : int
        Number of oscillons to place.
    N_grid : int
        Grid points per axis.
    L : float
        Box size.
    phi0 : float
        Peak amplitude of each oscillon.
    R : float
        Gaussian width parameter.
    min_separation : float
        Minimum distance between any pair of oscillon centers.
    seed : int or None
        Random seed for reproducibility.
    max_retries : int
        Maximum rejection sampling attempts per oscillon (default 1000).

    Returns
    -------
    phi : ndarray (N_grid, N_grid, N_grid)
        Initial field.
    phi_dot : ndarray (N_grid, N_grid, N_grid)
        Initial time derivative (zeros).
    config : dict
        Initial configuration with 'positions' and 'phases'.
    """
    rng = np.random.default_rng(seed)

    # Build grid coordinates (matching SexticEvolver convention)
    coords = np.linspace(-L / 2, L / 2, N_grid, endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

    # Place oscillons with rejection sampling
    margin = R  # keep centers away from box edges for cleaner periodicity
    lo = -L / 2 + margin
    hi = L / 2 - margin

    positions = []
    for i in range(N_osc):
        for attempt in range(max_retries):
            pos = rng.uniform(lo, hi, size=3)
            # Check distance to all previously placed oscillons
            too_close = False
            for prev in positions:
                dist = np.linalg.norm(pos - prev)
                if dist < min_separation:
                    too_close = True
                    break
            if not too_close:
                positions.append(pos)
                break
        else:
            raise RuntimeError(
                "Could not place oscillon %d after %d retries. "
                "Box too small or min_separation too large." % (i, max_retries))

    positions = np.array(positions)

    # Random phases: +1 or -1 with equal probability
    phases = rng.choice([-1, 1], size=N_osc)

    # Build superposed field
    phi = np.zeros((N_grid, N_grid, N_grid))
    for idx in range(N_osc):
        dx = X - positions[idx, 0]
        dy = Y - positions[idx, 1]
        dz = Z - positions[idx, 2]
        r2 = dx**2 + dy**2 + dz**2
        phi += phases[idx] * phi0 * np.exp(-r2 / (2.0 * R**2))

    phi_dot = np.zeros_like(phi)

    config = {
        "positions": positions.tolist(),
        "phases": phases.tolist(),
    }

    return phi, phi_dot, config
