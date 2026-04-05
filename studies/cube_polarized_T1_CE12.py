#!/usr/bin/env python3
"""
Run the CORRECT cube polarized T1 configuration at N=64.

The true T1/T2 bipartition of a cube assigns vertices to two interlocking
tetrahedra: A={0,3,5,6} and B={1,2,4,7} (using the vertex ordering from
studies/02a_cube_phase2.py). This gives ALL 12 cube edges as cross-edges
(f=1.000), since every edge connects an A-vertex to a B-vertex.

Phases: [-1, 1, 1, -1, 1, -1, -1, 1]  (A at pi, B at 0)

The existing polarized_T1.json in outputs/phase2/cube/ used a DIFFERENT
4-flip assignment that gives only CE=8 (f=0.667). This script produces
the missing f=1.000 data point.

Uses run_with_checkpointing() from engine/checkpoint.py.
"""

import json
import os
import sys
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.evolver import SexticEvolver
from engine.checkpoint import run_with_checkpointing

# -- Parameters (Set B) -------------------------------------------------------
N_GRID = 64
L      = 50.0
m      = 1.0
g4     = 0.30
g6     = 0.055
dt     = 0.05
sigma  = 0.01
phi0   = 0.5
R      = 2.5
T_FINAL = 500.0
D_EDGE  = 6.0

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
PHASE1_JSON = os.path.join(BASE_DIR, "outputs", "phase1", "set_B_baseline.json")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "phase2", "cube")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cube_polarized_T1_CE12.json")

# -- Cube geometry (same vertex ordering as 02a_cube_phase2.py) ----------------
D = D_EDGE / 2.0  # 3.0
VERTS = np.array([
    [-D, -D, -D],  # 0
    [ D, -D, -D],  # 1
    [-D,  D, -D],  # 2
    [ D,  D, -D],  # 3
    [-D, -D,  D],  # 4
    [ D, -D,  D],  # 5
    [-D,  D,  D],  # 6
    [ D,  D,  D],  # 7
])

EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if np.linalg.norm(VERTS[i] - VERTS[j]) < D_EDGE + 0.1:
            EDGES.append((i, j))

# Correct T1/T2 bipartition: A={0,3,5,6} at pi, B={1,2,4,7} at 0
PHASES = [-1, 1, 1, -1, 1, -1, -1, 1]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verify CE=12 before running
    n_cross = sum(1 for (i, j) in EDGES if PHASES[i] != PHASES[j])
    print("=" * 60)
    print("  Cube Polarized T1 (correct bipartition)")
    print("  Phases: %s" % PHASES)
    print("  Vertices at pi: %s" % sorted(i for i, p in enumerate(PHASES) if p < 0))
    print("  Vertices at 0:  %s" % sorted(i for i, p in enumerate(PHASES) if p > 0))
    print("  Edges: %s" % EDGES)
    print("  Cross-edges: %d / %d (f=%.3f)" % (n_cross, len(EDGES), n_cross / len(EDGES)))
    print("=" * 60)
    sys.stdout.flush()

    assert n_cross == 12, "Expected CE=12, got CE=%d" % n_cross
    assert len(EDGES) == 12, "Expected 12 edges, got %d" % len(EDGES)

    # Build evolver and IC
    ev = SexticEvolver(N=N_GRID, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    phi_init = np.zeros((N_GRID, N_GRID, N_GRID))
    for A, pos in zip(PHASES, VERTS):
        dx_ = ev.X - pos[0]
        dy_ = ev.Y - pos[1]
        dz_ = ev.Z - pos[2]
        r2 = dx_**2 + dy_**2 + dz_**2
        phi_init += A * phi0 * np.exp(-r2 / (2.0 * R**2))

    phi_dot_init = np.zeros_like(phi_init)
    ev.set_initial_conditions(phi_init, phi_dot_init)

    E0 = ev.compute_energy()
    print("  E(0) = %.6e  max|phi|(0) = %.6f" % (E0, float(np.max(np.abs(phi_init)))))
    sys.stdout.flush()

    # Config dict for run_with_checkpointing
    config = {
        'name': 'cube_polarized_T1_CE12',
        'params': {
            'N_grid': N_GRID,
            'L': L,
            'm': m,
            'g4': g4,
            'g6': g6,
            'dt': dt,
            'T_final': T_FINAL,
            'sigma_KO': sigma,
            'd_edge': D_EDGE,
        },
        'initial_conditions': {
            'geometry': 'cube',
            'n_oscillons': 8,
            'phases': PHASES,
            'cross_edges': 12,
            'f_cross': 1.0,
            'phi0': phi0,
            'R': R,
        },
        'metadata': {
            'geometry': 'cube_polarized_T1_CE12',
            'description': 'Correct T1/T2 bipartition: A={0,3,5,6} at pi, B={1,2,4,7} at 0',
        },
        'record_every': 10,
        'print_every': 2000,
    }

    results = run_with_checkpointing(ev, config, OUTPUT_PATH)

    # Post-process: compute E_bind
    with open(PHASE1_JSON) as f:
        ref = json.load(f)
    E_single_interp = interp1d(ref["t_series"], ref["E_series"],
                               kind="linear", fill_value="extrapolate")

    times = results['time_series']['times']
    E_total = results['time_series']['E_total']

    E_bind_series = [
        E_total[i] - 8.0 * float(E_single_interp(times[i]))
        for i in range(len(times))
    ]

    Ebind_final = E_bind_series[-1]
    E_final = E_total[-1]
    drift_final = abs(E_final - E_total[0]) / (abs(E_total[0]) + 1e-30)
    max_amp = results['time_series']['max_amplitude']
    amp_ret = max_amp[-1] / (phi0 * 1.0)
    verdict = "STABLE" if Ebind_final < -1.0 else "UNSTABLE"

    # Append E_bind data to the output file
    with open(OUTPUT_PATH) as f:
        data = json.load(f)

    data['cross_edges'] = 12
    data['n_pi'] = sum(1 for p in PHASES if p < 0)
    data['phases'] = PHASES
    data['E_bind_series'] = E_bind_series
    data['final_E_bind'] = Ebind_final
    data['energy_drift_final'] = drift_final
    data['amplitude_retention_final'] = amp_ret
    data['verdict'] = verdict

    tmp_path = OUTPUT_PATH + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    os.replace(tmp_path, OUTPUT_PATH)

    print()
    print("=" * 60)
    print("  RESULT: Cube Polarized T1 (CE=12, f=1.000)")
    print("  E_bind(T=500) = %.6f" % Ebind_final)
    print("  Verdict: %s" % verdict)
    print("  Amplitude retention: %.4f" % amp_ret)
    print("  Energy drift: %.2e" % drift_final)
    print("  Saved -> %s" % OUTPUT_PATH)
    print("=" * 60)


if __name__ == '__main__':
    main()
