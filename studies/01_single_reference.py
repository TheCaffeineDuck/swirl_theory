"""
studies/01_single_reference.py
==============================
Phase 1: Single Oscillon Baseline -- Set B

Evolves a single isolated Gaussian oscillon to T=500 using Set B parameters,
producing the time-matched E_single(t) reference for all subsequent phases.

Set B: phi0=0.5, R=2.5
Fixed: m=1.0, g4=0.30, g6=0.055, N=64, L=50.0, dt=0.05, sigma=0.01

Features incremental checkpointing every 1000 steps (~50 time units).
"""

import sys
import os
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.evolver import SexticEvolver

# -- Parameters ---------------------------------------------------------------
N    = 64
L    = 50.0
m    = 1.0
g4   = 0.30
g6   = 0.055
dt   = 0.05
sigma = 0.01
phi0 = 0.5
R    = 2.5

T_END        = 500.0
N_STEPS      = int(T_END / dt)   # 10000
RECORD_EVERY = 10                 # every 10 steps = 0.5 time units
PRINT_EVERY  = 1000               # every 1000 steps = 50 time units
CHECKPOINT_EVERY = 1000

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_PATH = Path(os.path.join(OUTPUT_DIR, "set_B_baseline.json"))
CHECKPOINT_PATH = OUT_PATH.with_suffix('.checkpoint.json')


# -- Checkpoint helpers -------------------------------------------------------
def checkpoint_callback(state):
    """Atomic incremental save."""
    tmp_path = str(CHECKPOINT_PATH) + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(state, f)
    os.replace(tmp_path, str(CHECKPOINT_PATH))


# -- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    # Skip if already completed
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            data = json.load(f)
        if data.get('completed', False):
            print("  CACHED: set_B_baseline")
            print("  Output: %s" % str(OUT_PATH))
            sys.exit(0)

    # Check for checkpoint to resume from
    resume_state = None
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            resume_state = json.load(f)
        print("  RESUMING from step %d" % resume_state['completed_steps'])

    # Setup
    ev = SexticEvolver(N=N, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    if resume_state is None:
        r2 = ev.X**2 + ev.Y**2 + ev.Z**2
        phi_init = phi0 * np.exp(-r2 / (2.0 * R**2))
        phi_dot_init = np.zeros_like(phi_init)
        ev.set_initial_conditions(phi_init, phi_dot_init)

        print("Set B: phi0=%.2f  R=%.1f" % (phi0, R))
        print("N=%d  L=%.1f  dt=%.3f  T_end=%.1f  steps=%d" % (N, L, dt, T_END, N_STEPS))
        print("E(0) = %.6e" % ev.compute_energy())
        print("max|phi|(0) = %.6f" % float(np.max(np.abs(phi_init))))
        print("")

    # Evolve with checkpointing
    state = ev.evolve(
        dt=dt, n_steps=N_STEPS, record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=checkpoint_callback,
        resume_from=resume_state,
        print_every=PRINT_EVERY,
        tag="setB_baseline",
    )

    # Build final result in existing schema
    ts = state['time_series']
    E0 = state['E0']
    E_final = ts['E_total'][-1]
    amp_final = ts['max_amplitude'][-1]
    drift_final = abs(E_final - E0) / (abs(E0) + 1e-30)
    amp_retention = amp_final / phi0

    result = {
        "param_set":               "B",
        "phi0":                    phi0,
        "R":                       R,
        "t_series":                ts['times'],
        "E_series":                ts['E_total'],
        "max_phi_series":          ts['max_amplitude'],
        "energy_drift_final":      drift_final,
        "amplitude_retention_final": amp_retention,
        "completed":               True,
    }

    # Atomic write
    tmp = str(OUT_PATH) + '.tmp'
    with open(tmp, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    os.replace(tmp, str(OUT_PATH))

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # Summary
    print("")
    print("=" * 55)
    print("  FINAL SUMMARY -- Set B")
    print("=" * 55)
    print("  Runtime              : %.1f s (%.2f min)" % (
        state['wall_elapsed'], state['wall_elapsed'] / 60))
    print("  E(0)                 : %.6e" % E0)
    print("  E(T=500)             : %.6e" % E_final)
    print("  Energy drift         : %.4e" % drift_final)
    print("  Final max|phi|       : %.6f" % amp_final)
    print("  Amplitude retention  : %.4f  (final/initial)" % amp_retention)
    print("  Records saved        : %d" % len(ts['times']))
    print("  Output               : %s" % str(OUT_PATH))
    print("=" * 55)
    print("  Phase 1 Set B complete.")
