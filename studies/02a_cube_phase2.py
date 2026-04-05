"""
studies/02a_cube_phase2.py
==========================
Phase 2: 8-oscillon cube arrangement
4 configurations varying phase pattern (cross-edge counts: 0, 3, 6, 12)
Binding energy E_bind(t) = E_total(t) - 8 * E_single(t)

Uses engine's evolve() with incremental checkpointing and resume support.
"""

import sys
import os
import json
import time

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.evolver import SexticEvolver, serialize_field, deserialize_field

# -- Parameters ---------------------------------------------------------------
N      = 64
L      = 50.0
m      = 1.0
g4     = 0.30
g6     = 0.055
dt     = 0.05
sigma  = 0.01
phi0   = 0.5
R      = 2.5
T_END  = 500.0
N_STEPS         = int(T_END / dt)   # 10000
RECORD_EVERY    = 10
PRINT_EVERY     = 1000
CHECKPOINT_EVERY = 1000

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
PHASE1_JSON = os.path.join(BASE_DIR, "outputs", "phase1", "set_B_baseline.json")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "phase2", "cube")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Cube vertices (edge length = 6.0) ----------------------------------------
# vertices indexed 0..7: (+-3, +-3, +-3)
D = 3.0
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

# Cube edges (all pairs with |vi-vj|=6)
EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if np.linalg.norm(VERTS[i] - VERTS[j]) < 6.1:
            EDGES.append((i, j))
# Should be 12 edges

def count_cross_edges(phases):
    return sum(1 for (i, j) in EDGES if phases[i] != phases[j])

# -- Configurations -----------------------------------------------------------
CONFIGS = [
    ("all_same_phase",  [ 1, 1, 1, 1, 1, 1, 1, 1]),  # 0 cross-edges
    ("single_flip",     [-1, 1, 1, 1, 1, 1, 1, 1]),  # 3 cross-edges
    ("checkerboard",    [-1,-1,-1, 1, 1, 1,-1, 1]),   # 6 cross-edges
    ("polarized_T1",    [-1, 1, 1,-1, 1,-1,-1, 1]),   # 12 cross-edges (bipartite: A={0,3,5,6}, B={1,2,4,7})
]

# -- Evolution routine --------------------------------------------------------
def run_config(name, phases):
    phases = np.array(phases, dtype=float)
    n_cross = count_cross_edges(phases)
    n_pi = int(np.sum(phases < 0))

    out_path = os.path.join(OUTPUT_DIR, "%s.json" % name)
    ckpt_path = os.path.join(OUTPUT_DIR, "%s.checkpoint.json" % name)

    # 1. Check if final output already exists and is complete -> skip
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if existing.get("completed"):
                print("")
                print("=" * 60)
                print("  Config: %s  -- CACHED (skipping)" % name)
                print("=" * 60)
                sys.stdout.flush()
                return {
                    "config_name":          name,
                    "cross_edges":          existing["cross_edges"],
                    "cross_edge_fraction":  existing["cross_edges"] / 12.0,
                    "final_E_bind":         existing["final_E_bind"],
                    "amplitude_retention":  existing["amplitude_retention_final"],
                    "energy_drift":         existing["energy_drift_final"],
                    "verdict":              existing["verdict"],
                }
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted file, re-run

    print("")
    print("=" * 60)
    print("  Config: %s" % name)
    print("  Cross-edges: %d / 12  (%.3f)" % (n_cross, n_cross / 12.0))
    print("  Pi-phases:   %d / 8" % n_pi)
    print("=" * 60)
    sys.stdout.flush()

    # Load E_single reference (inside run_config, not at module level)
    with open(PHASE1_JSON) as f:
        ref = json.load(f)
    E_single_interp = interp1d(ref["t_series"], ref["E_series"],
                               kind="linear", fill_value="extrapolate")

    ev = SexticEvolver(N=N, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    # Build initial conditions: sum of 8 Gaussians
    phi_init = np.zeros((N, N, N))
    for idx, (A, pos) in enumerate(zip(phases, VERTS)):
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

    # 2. Check for checkpoint -> resume
    resume_state = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_state = json.load(f)
            print("  Found checkpoint at step %d (t=%.1f)" % (
                resume_state['completed_steps'], resume_state['t']))
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            print("  Corrupted checkpoint, starting fresh")
            sys.stdout.flush()
            resume_state = None

    # 3. Checkpoint callback: atomic write via .tmp + os.replace
    def checkpoint_callback(state_dict):
        tmp_path = ckpt_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state_dict, f, separators=(",", ":"))
        os.replace(tmp_path, ckpt_path)

    # 4. Run evolve()
    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=checkpoint_callback,
        resume_from=resume_state,
        print_every=PRINT_EVERY,
        tag=name,
    )

    times = state['time_series']['times']
    E_total = state['time_series']['E_total']
    max_amp = state['time_series']['max_amplitude']
    wall_time = state['wall_elapsed']

    # 5. Compute E_bind series post-hoc
    E_bind_series = [
        E_total[i] - 8.0 * float(E_single_interp(times[i]))
        for i in range(len(times))
    ]

    E_final      = E_total[-1]
    Ebind_final  = E_bind_series[-1]
    amp_final    = max_amp[-1]
    drift_final  = abs(E_final - state['E0']) / (abs(state['E0']) + 1e-30)
    amp_ret      = amp_final / (phi0 * max(1, abs(phases).sum() / len(phases)))
    verdict      = "STABLE" if Ebind_final < -1.0 else "UNSTABLE"

    print("")
    print("  -- %s complete (%.1f s) --" % (name, wall_time))
    print("  E_bind(T) = %.4e  |  verdict: %s" % (Ebind_final, verdict))
    sys.stdout.flush()

    # 6. Save final result
    per_config = {
        "config_name":          name,
        "cross_edges":          n_cross,
        "n_pi":                 n_pi,
        "phases":               list(phases),
        "t_series":             times,
        "E_series":             E_total,
        "E_bind_series":        E_bind_series,
        "max_phi_series":       max_amp,
        "energy_drift_final":   drift_final,
        "amplitude_retention_final": amp_ret,
        "final_E_bind":         Ebind_final,
        "verdict":              verdict,
        "runtime_seconds":      wall_time,
        "completed":            True,
    }
    with open(out_path, "w") as f:
        json.dump(per_config, f, separators=(",", ":"))
    print("  Saved -> %s" % out_path)
    sys.stdout.flush()

    # 7. Delete checkpoint now that final output is saved
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return {
        "config_name":          name,
        "cross_edges":          n_cross,
        "cross_edge_fraction":  n_cross / 12.0,
        "final_E_bind":         Ebind_final,
        "amplitude_retention":  amp_ret,
        "energy_drift":         drift_final,
        "verdict":              verdict,
    }

# -- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    print("02a_cube_phase2.py  --  4 configurations")
    print("Output dir: %s" % os.path.abspath(OUTPUT_DIR))
    sys.stdout.flush()

    summaries = []
    for name, phases in CONFIGS:
        summaries.append(run_config(name, phases))

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print("\nSummary saved -> %s" % summary_path)

    # Print summary table
    print("")
    print("=" * 90)
    print("  CUBE SUMMARY TABLE")
    print("=" * 90)
    print("  %-20s %8s %10s %14s %12s %10s %10s" % (
        "Config", "X-edges", "X-frac", "E_bind(T)", "Amp Ret", "Drift", "Verdict"))
    print("  " + "-" * 86)
    for s in summaries:
        print("  %-20s %8d %10.3f %14.4e %12.4f %10.3e %10s" % (
            s["config_name"], s["cross_edges"], s["cross_edge_fraction"],
            s["final_E_bind"], s["amplitude_retention"],
            s["energy_drift"], s["verdict"]))
    print("=" * 90)
    print("  Phase 2 Cube complete.")
