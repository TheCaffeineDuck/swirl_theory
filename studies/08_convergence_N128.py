#!/usr/bin/env python3
"""
studies/08_convergence_N128.py
==============================
Phase 4 Convergence Check: Re-run 3 configs at N=128 to verify N=64 results.

Runs in parallel using multiprocessing.Pool(3):
  1. Isolated baseline (N=128) -- needed as time-matched reference
  2. Cube polarized T1 (N=128) -- most clearly bound config
  3. Tetrahedron single-flip (N=128) -- threshold config

Features:
  - Checkpoints every 1000 steps via evolve() (base64 fields in JSON)
  - Auto-resumes from .checkpoint.json on restart
  - Final outputs saved with completed=True

Compares E_bind at N=64 vs N=128.
Pass criterion: E_bind agrees within 10% (or sign agreement for near-zero values).

See: STUDY_PLAN_2026-03-18.md (Phase 4) and COMPUTE_GUIDELINES.md
"""

import sys
import os
import json
import time
import multiprocessing
from datetime import datetime

# Prevent Numba thread oversubscription
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np

# -- Paths --------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PHASE1_JSON = os.path.join(BASE_DIR, "outputs", "phase1", "set_B_baseline.json")
CUBE_N64_JSON = os.path.join(BASE_DIR, "outputs", "phase2", "cube", "polarized_T1.json")
TET_N64_JSON = os.path.join(BASE_DIR, "outputs", "phase4_new_geometries",
                             "tetrahedron", "single_flip.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "convergence")

# -- Parameters (Set B, same as all prior runs except N_grid) -----------------
N_GRID = 128
L      = 50.0
m      = 1.0
g4     = 0.30
g6     = 0.055
dt     = 0.05
sigma  = 0.01
phi0   = 0.5
R      = 2.5
T_END  = 500.0
N_STEPS      = int(T_END / dt)   # 10000
RECORD_EVERY = 10
PRINT_EVERY  = 1000
CHECKPOINT_EVERY = 1000


# =============================================================================
#  GEOMETRY DEFINITIONS (same as existing studies)
# =============================================================================

def make_cube_vertices():
    """Cube vertices at (+-3, +-3, +-3), edge length = 6.0."""
    D = 3.0
    verts = np.array([
        [-D, -D, -D],  # 0
        [ D, -D, -D],  # 1
        [-D,  D, -D],  # 2
        [ D,  D, -D],  # 3
        [-D, -D,  D],  # 4
        [ D, -D,  D],  # 5
        [-D,  D,  D],  # 6
        [ D,  D,  D],  # 7
    ])
    return verts


def make_tetrahedron_vertices():
    """Regular tetrahedron, edge length = 6.0."""
    base = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float)
    current_edge = 2.0 * np.sqrt(2.0)
    scale = 6.0 / current_edge
    return base * scale


# =============================================================================
#  WORKER FUNCTION
# =============================================================================

def run_simulation(args):
    """Run a single N=128 simulation with checkpointing. Called by Pool workers."""
    config_name, n_oscillons, verts, phases = args

    # Import engine inside worker (for multiprocessing spawn)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from engine.evolver import SexticEvolver

    output_path = os.path.join(OUTPUT_DIR, "%s.json" % config_name)
    checkpoint_path = os.path.join(OUTPUT_DIR, "%s.checkpoint.json" % config_name)

    # Check if final output exists with completed=True -> skip
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if existing.get("completed", False):
                print("  [%s] CACHED -- skipping" % config_name)
                sys.stdout.flush()
                return existing
        except (json.JSONDecodeError, KeyError):
            pass

    # Check for checkpoint to resume from
    resume_from = None
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                resume_from = json.load(f)
            print("  [%s] Resuming from checkpoint (step %d)" % (
                config_name, resume_from['completed_steps']))
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            resume_from = None

    tag = config_name
    phases_arr = np.array(phases, dtype=float)

    ev = SexticEvolver(N=N_GRID, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    # Build initial conditions
    print("")
    print("=" * 60)
    if resume_from is not None:
        print("  [%s] RESUMING N=%d simulation" % (tag, N_GRID))
    else:
        print("  [%s] Starting N=%d simulation (fresh)" % (tag, N_GRID))
    print("  Oscillons: %d" % n_oscillons)
    if verts is not None and resume_from is None:
        print("  Phases: %s" % list(phases))
    print("=" * 60)
    sys.stdout.flush()

    phi_init = np.zeros((N_GRID, N_GRID, N_GRID))
    if verts is None:
        r2 = ev.X**2 + ev.Y**2 + ev.Z**2
        phi_init = phi0 * np.exp(-r2 / (2.0 * R**2))
    else:
        for idx in range(n_oscillons):
            pos = verts[idx]
            dx_ = ev.X - pos[0]
            dy_ = ev.Y - pos[1]
            dz_ = ev.Z - pos[2]
            r2 = dx_**2 + dy_**2 + dz_**2
            phi_init += phases_arr[idx] * phi0 * np.exp(-r2 / (2.0 * R**2))

    phi_dot_init = np.zeros_like(phi_init)
    ev.set_initial_conditions(phi_init, phi_dot_init)

    if resume_from is None:
        E0 = ev.compute_energy()
        max_phi0 = float(np.max(np.abs(phi_init)))
        print("  [%s] E(0) = %.6e  max|phi|(0) = %.6f" % (tag, E0, max_phi0))
        sys.stdout.flush()

    # Checkpoint callback: atomic write
    def checkpoint_cb(state):
        tmp_path = checkpoint_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(state, f)
        os.replace(tmp_path, checkpoint_path)

    # Run evolution
    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=checkpoint_cb,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=tag,
    )

    wall_time = state['wall_elapsed']
    E0 = state['E0']
    max_phi0 = state['max_phi0']
    t_series = state['time_series']['times']
    E_series = state['time_series']['E_total']
    max_phi_series = state['time_series']['max_amplitude']

    E_final = E_series[-1]
    drift_final = abs(E_final - E0) / (abs(E0) + 1e-30)
    amp_ret = max_phi_series[-1] / max_phi0 if max_phi0 > 0 else 0.0

    print("")
    print("  [%s] COMPLETE in %.1f s (%.1f min)" % (tag, wall_time, wall_time / 60.0))
    print("  E(0)=%.6e  E(T)=%.6e  drift=%.2e" % (E0, E_final, drift_final))
    sys.stdout.flush()

    # Build result dict
    result = {
        "completed": True,
        "metadata": {
            "geometry": config_name,
            "config_name": config_name,
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": wall_time,
        },
        "parameters": {
            "phi0": phi0,
            "R": R,
            "m": m,
            "g4": g4,
            "g6": g6,
            "N_grid": N_GRID,
            "L": L,
            "dt": dt,
            "T_final": T_END,
            "sigma_KO": sigma,
            "d_edge": 6.0,
        },
        "initial_conditions": {
            "n_oscillons": n_oscillons,
            "phases": list(phases) if phases is not None else [1],
        },
        "time_series": {
            "times": t_series,
            "E_total": E_series,
            "max_amplitude": max_phi_series,
        },
        "final_state": {
            "E_total_0": E0,
            "E_total_final": E_final,
            "max_amplitude_0": max_phi0,
            "max_amplitude_final": max_phi_series[-1],
            "amplitude_retention": amp_ret,
            "energy_drift_pct": drift_final,
        },
        "runtime_seconds": wall_time,
    }

    # Atomic write of final result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_path = output_path + '.tmp'
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, output_path)
    print("  Saved -> %s" % output_path)
    sys.stdout.flush()

    # Delete checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return result


# =============================================================================
#  MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wall_start_total = time.perf_counter()

    print("=" * 65)
    print("  Phase 4: Convergence Check (N=128)")
    print("  3 simulations in parallel using Pool(3)")
    print("  Checkpoints every %d steps" % CHECKPOINT_EVERY)
    print("  Output dir: %s" % os.path.abspath(OUTPUT_DIR))
    print("=" * 65)
    print("")

    # Load N=64 reference data for comparison
    print("Loading N=64 reference data...")

    with open(PHASE1_JSON) as f:
        baseline_n64 = json.load(f)
    E_single_0_N64 = baseline_n64["E_series"][0]
    E_single_500_N64 = baseline_n64["E_series"][-1]
    print("  Baseline N=64: E(0)=%.6f  E(500)=%.6f" % (E_single_0_N64, E_single_500_N64))

    with open(CUBE_N64_JSON) as f:
        cube_n64 = json.load(f)
    E_bind_cube_N64 = cube_n64["final_E_bind"]
    print("  Cube polarized T1 N=64: E_bind=%.4f" % E_bind_cube_N64)

    with open(TET_N64_JSON) as f:
        tet_n64 = json.load(f)
    E_bind_tet_N64 = tet_n64["final_E_bind"]
    print("  Tet single-flip N=64: E_bind=%.4f" % E_bind_tet_N64)
    print("")

    # Check for existing completions / checkpoints
    for name in ["isolated_N128", "cube_polarized_N128", "tet_1flip_N128"]:
        out_path = os.path.join(OUTPUT_DIR, "%s.json" % name)
        ckpt_path = os.path.join(OUTPUT_DIR, "%s.checkpoint.json" % name)
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    data = json.load(f)
                if data.get("completed", False):
                    print("  [%s] already complete -- will skip" % name)
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path) as f:
                    ckpt = json.load(f)
                print("  [%s] checkpoint found at step %d -- will resume" % (
                    name, ckpt['completed_steps']))
                continue
            except (json.JSONDecodeError, KeyError):
                pass
        print("  [%s] no checkpoint -- will start fresh" % name)
    print("")

    # Geometry definitions
    cube_verts = make_cube_vertices()
    tet_verts = make_tetrahedron_vertices()

    # Phase assignments (must match the existing N=64 runs exactly)
    cube_phases = [-1, 1, 1, -1, 1, -1, -1, 1]
    tet_phases = [-1, 1, 1, 1]

    # Build work items: (config_name, n_oscillons, verts, phases)
    work = [
        ("isolated_N128", 1, None, [1]),
        ("cube_polarized_N128", 8, cube_verts, cube_phases),
        ("tet_1flip_N128", 4, tet_verts, tet_phases),
    ]

    print("Running 3 configs with Pool(3)...")
    print("  Memory estimate: ~200 MB each, ~600 MB total")
    print("")
    sys.stdout.flush()

    with multiprocessing.Pool(3) as pool:
        results = pool.map(run_simulation, work)

    # Unpack results
    baseline_n128 = results[0]
    cube_n128 = results[1]
    tet_n128 = results[2]

    # Extract N=128 baseline energies for binding energy computation
    from scipy.interpolate import interp1d

    E_single_interp_N128 = interp1d(
        baseline_n128["time_series"]["times"],
        baseline_n128["time_series"]["E_total"],
        kind="linear", fill_value="extrapolate"
    )

    # Compute binding energies at N=128
    # Cube: E_bind = E_total - 8 * E_single
    cube_times = cube_n128["time_series"]["times"]
    cube_E_total = cube_n128["time_series"]["E_total"]
    cube_E_bind_series = [
        cube_E_total[i] - 8.0 * float(E_single_interp_N128(cube_times[i]))
        for i in range(len(cube_times))
    ]
    E_bind_cube_N128 = cube_E_bind_series[-1]

    # Tet: E_bind = E_total - 4 * E_single
    tet_times = tet_n128["time_series"]["times"]
    tet_E_total = tet_n128["time_series"]["E_total"]
    tet_E_bind_series = [
        tet_E_total[i] - 4.0 * float(E_single_interp_N128(tet_times[i]))
        for i in range(len(tet_times))
    ]
    E_bind_tet_N128 = tet_E_bind_series[-1]

    # Update saved results with binding energy
    cube_n128["time_series"]["E_bind"] = cube_E_bind_series
    cube_n128["final_state"]["E_bind_final"] = E_bind_cube_N128
    tmp_path = os.path.join(OUTPUT_DIR, "cube_polarized_N128.json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(cube_n128, f, indent=2)
    os.replace(tmp_path, os.path.join(OUTPUT_DIR, "cube_polarized_N128.json"))

    tet_n128["time_series"]["E_bind"] = tet_E_bind_series
    tet_n128["final_state"]["E_bind_final"] = E_bind_tet_N128
    tmp_path = os.path.join(OUTPUT_DIR, "tet_1flip_N128.json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(tet_n128, f, indent=2)
    os.replace(tmp_path, os.path.join(OUTPUT_DIR, "tet_1flip_N128.json"))

    # N=128 baseline values
    E_single_0_N128 = baseline_n128["time_series"]["E_total"][0]
    E_single_500_N128 = baseline_n128["time_series"]["E_total"][-1]

    # Compute relative differences
    def rel_diff(a, b):
        avg = 0.5 * (abs(a) + abs(b))
        if avg < 1e-12:
            return 0.0
        return abs(a - b) / avg

    rel_diff_E0 = rel_diff(E_single_0_N64, E_single_0_N128)
    rel_diff_E500 = rel_diff(E_single_500_N64, E_single_500_N128)

    # Cube convergence
    rel_diff_cube = rel_diff(E_bind_cube_N64, E_bind_cube_N128)
    cube_pass = rel_diff_cube < 0.10
    cube_sign_agrees = (E_bind_cube_N64 < 0) == (E_bind_cube_N128 < 0)

    # Tetrahedron convergence (special case: near-zero)
    tet_sign_agrees = (E_bind_tet_N64 < 0) == (E_bind_tet_N128 < 0)
    tet_near_zero = abs(E_bind_tet_N128) < 1.0
    tet_pass = tet_sign_agrees and tet_near_zero
    rel_diff_tet = rel_diff(E_bind_tet_N64, E_bind_tet_N128)

    # Overall verdict
    overall_pass = cube_pass and tet_pass

    wall_total = time.perf_counter() - wall_start_total

    # Save convergence summary
    summary = {
        "study": "convergence_check",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "configs_tested": [
            {
                "geometry": "cube",
                "config": "polarized_T1",
                "E_bind_N64": E_bind_cube_N64,
                "E_bind_N128": E_bind_cube_N128,
                "relative_difference": rel_diff_cube,
                "sign_agrees": cube_sign_agrees,
                "pass_10pct": cube_pass,
            },
            {
                "geometry": "tetrahedron",
                "config": "single_flip",
                "E_bind_N64": E_bind_tet_N64,
                "E_bind_N128": E_bind_tet_N128,
                "relative_difference": rel_diff_tet,
                "sign_agrees": tet_sign_agrees,
                "note": "Near-zero E_bind; sign agreement is the meaningful test",
                "pass": tet_pass,
            },
        ],
        "baseline_comparison": {
            "E_single_0_N64": E_single_0_N64,
            "E_single_0_N128": E_single_0_N128,
            "E_single_500_N64": E_single_500_N64,
            "E_single_500_N128": E_single_500_N128,
            "relative_difference_E0": rel_diff_E0,
            "relative_difference_E500": rel_diff_E500,
        },
        "overall_verdict": "PASS" if overall_pass else "FAIL",
        "wall_clock_minutes": wall_total / 60.0,
    }

    summary_path = os.path.join(OUTPUT_DIR, "convergence_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("")
    print("Saved convergence summary -> %s" % summary_path)

    # Print summary
    print("")
    print("=" * 65)
    print("  CONVERGENCE CHECK COMPLETE")
    print("=" * 65)
    print("")
    print("  Isolated baseline:")
    print("    E_single(0):   N=64: %.6f   N=128: %.6f   diff: %.2f%%" % (
        E_single_0_N64, E_single_0_N128, rel_diff_E0 * 100))
    print("    E_single(500): N=64: %.6f   N=128: %.6f   diff: %.2f%%" % (
        E_single_500_N64, E_single_500_N128, rel_diff_E500 * 100))
    print("")
    print("  Cube polarized T1:")
    print("    E_bind: N=64: %.4f   N=128: %.4f   diff: %.2f%%" % (
        E_bind_cube_N64, E_bind_cube_N128, rel_diff_cube * 100))
    print("    %s" % ("PASS" if cube_pass else "FAIL"))
    print("")
    print("  Tetrahedron single-flip:")
    print("    E_bind: N=64: %.4f   N=128: %.4f" % (
        E_bind_tet_N64, E_bind_tet_N128))
    print("    sign agrees: %s" % ("YES" if tet_sign_agrees else "NO"))
    print("    %s" % ("PASS" if tet_pass else "FAIL"))
    print("")
    print("  Overall: %s" % ("PASS" if overall_pass else "FAIL"))
    print("  Wall-clock time: %.1f min" % (wall_total / 60.0))
    print("=" * 65)


if __name__ == "__main__":
    main()
