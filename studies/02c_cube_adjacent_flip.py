"""
studies/02c_cube_adjacent_flip.py
==================================
Phase 2 supplement: Run a single cube config "adjacent_flip" with
vertices 0 (T1) and 1 (T2) at phase pi, all others at phase 0.
This gives exactly 4/12 cross-edges (cross_edge_fraction = 0.333).

Adds the result to outputs/phase2/cube/summary.json.

Uses engine's evolve() with incremental checkpointing and resume.
"""

import sys
import os
import json
import tempfile

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.evolver import SexticEvolver

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

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
PHASE1_JSON = os.path.join(BASE_DIR, "outputs", "phase1", "set_B_baseline.json")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "phase2", "cube")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Cube vertices (edge length = 6.0) ----------------------------------------
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

EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if np.linalg.norm(VERTS[i] - VERTS[j]) < 6.1:
            EDGES.append((i, j))

def count_cross_edges(phases):
    return sum(1 for (i, j) in EDGES if phases[i] != phases[j])


def run_config(name, phases):
    phases = np.array(phases, dtype=float)
    n_cross = count_cross_edges(phases)
    n_pi = int(np.sum(phases < 0))

    out_path = os.path.join(OUTPUT_DIR, "%s.json" % name)
    ckpt_path = out_path + ".checkpoint.json"

    # -- Check for cached final result ----------------------------------------
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("completed"):
                print("")
                print("  Config: %s  -> CACHED, skipping." % name)
                sys.stdout.flush()
                return {
                    "config_name":          name,
                    "cross_edges":          cached["cross_edges"],
                    "cross_edge_fraction":  cached["cross_edges"] / 12.0,
                    "final_E_bind":         cached["final_E_bind"],
                    "amplitude_retention":  cached["amplitude_retention_final"],
                    "energy_drift":         cached["energy_drift_final"],
                    "verdict":              cached["verdict"],
                }
        except (json.JSONDecodeError, KeyError):
            pass

    # -- Load baseline (inside run_config, not module level) ------------------
    with open(PHASE1_JSON) as f:
        ref = json.load(f)
    E_single_interp = interp1d(ref["t_series"], ref["E_series"],
                               kind="linear", fill_value="extrapolate")

    print("")
    print("=" * 60)
    print("  Config: %s" % name)
    print("  Cross-edges: %d / 12  (%.3f)" % (n_cross, n_cross / 12.0))
    print("  Pi-phases:   %d / 8" % n_pi)
    print("=" * 60)
    sys.stdout.flush()

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

    # -- Check for checkpoint to resume from ----------------------------------
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  Found checkpoint at step %d (t=%.1f)" % (
                resume_from['completed_steps'], resume_from['t']))
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            resume_from = None

    # -- Checkpoint callback (atomic write) -----------------------------------
    def save_checkpoint(state_dict):
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=OUTPUT_DIR, suffix=".tmp", prefix=".ckpt_")
        try:
            with os.fdopen(tmp_fd, "w") as tmp_f:
                json.dump(state_dict, tmp_f, separators=(",", ":"))
            os.replace(tmp_path, ckpt_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        print("  [checkpoint] step=%d  t=%.1f" % (
            state_dict['completed_steps'], state_dict['t']))
        sys.stdout.flush()

    # -- Run evolution --------------------------------------------------------
    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=save_checkpoint,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=name,
    )

    # -- Compute E_bind post-hoc ----------------------------------------------
    times = state['time_series']['times']
    E_total = state['time_series']['E_total']
    max_amplitude = state['time_series']['max_amplitude']

    Ebind_series = []
    for i, t_i in enumerate(times):
        e_ref = float(E_single_interp(t_i))
        Ebind_series.append(E_total[i] - 8.0 * e_ref)

    E_final      = E_total[-1]
    Ebind_final  = Ebind_series[-1]
    amp_final    = max_amplitude[-1]
    drift_final  = abs(E_final - state['E0']) / (abs(state['E0']) + 1e-30)
    amp_ret      = amp_final / (phi0 * max(1, abs(phases).sum() / len(phases)))
    verdict      = "STABLE" if Ebind_final < -1.0 else "UNSTABLE"

    wall_time = state['wall_elapsed']

    print("")
    print("  -- %s complete (%.1f s) --" % (name, wall_time))
    print("  E_bind(T) = %.4e  |  verdict: %s" % (Ebind_final, verdict))
    sys.stdout.flush()

    # -- Save final result ----------------------------------------------------
    per_config = {
        "config_name":          name,
        "cross_edges":          n_cross,
        "n_pi":                 n_pi,
        "phases":               list(phases),
        "t_series":             list(times),
        "E_series":             list(E_total),
        "E_bind_series":        Ebind_series,
        "max_phi_series":       list(max_amplitude),
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

    # -- Delete checkpoint on success -----------------------------------------
    if os.path.exists(ckpt_path):
        os.unlink(ckpt_path)
        print("  Removed checkpoint -> %s" % ckpt_path)
        sys.stdout.flush()

    return {
        "config_name":          name,
        "cross_edges":          n_cross,
        "cross_edge_fraction":  n_cross / 12.0,
        "final_E_bind":         Ebind_final,
        "amplitude_retention":  amp_ret,
        "energy_drift":         drift_final,
        "verdict":              verdict,
    }


if __name__ == "__main__":
    print("02c_cube_adjacent_flip.py -- single config: adjacent_flip")
    print("Output dir: %s" % os.path.abspath(OUTPUT_DIR))
    sys.stdout.flush()

    # adjacent_flip: vertices 0 (T1) and 1 (T2) at phase pi
    # Gives 4 cross-edges / 12 = 0.333
    result = run_config("adjacent_flip", [-1, -1, +1, +1, +1, +1, +1, +1])

    # Update summary.json
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path) as f:
        summaries = json.load(f)

    # Remove any existing adjacent_flip entry
    summaries = [s for s in summaries if s["config_name"] != "adjacent_flip"]
    summaries.append(result)
    # Sort by cross_edge_fraction
    summaries.sort(key=lambda s: s["cross_edge_fraction"])

    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print("\nUpdated summary -> %s" % summary_path)

    # Print table
    print("")
    print("=" * 90)
    print("  UPDATED CUBE SUMMARY TABLE")
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
