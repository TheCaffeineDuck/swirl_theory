#!/usr/bin/env python3
"""
studies/09_selection_rule_N128.py
=================================
Phase 4b: Selection Rule Convergence Test

Runs 4 configs at N=128 to test whether the SIGN of E_bind is preserved:
  1. Cube all-same-phase    (E_bind_N64 = +73.84, should stay positive)
  2. Cube checkerboard      (E_bind_N64 = -3.86,  KEY TEST - marginally bound)
  3. Icosahedron ce_15      (E_bind_N64 = +3.56,  KEY TEST - broke 50% claim)
  4. Icosahedron ce_20      (E_bind_N64 = -50.11, strongest ico binding)

Uses existing N=128 isolated baseline from outputs/convergence/isolated_N128.json.
All 4 run in parallel via multiprocessing.Pool(4).

See: STUDY_PLAN_2026-03-18.md (Phase 4b)
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
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASELINE_N128_JSON = os.path.join(BASE_DIR, "outputs", "convergence", "isolated_N128.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "convergence")

# -- Parameters (Set B, N=128) ------------------------------------------------
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
PRINT_EVERY  = 2000
CHECKPOINT_EVERY = 1000


# =============================================================================
#  GEOMETRY DEFINITIONS
# =============================================================================

def make_cube_vertices():
    """Cube vertices at (+-3, +-3, +-3), edge length = 6.0."""
    D = 3.0
    return np.array([
        [-D, -D, -D],  # 0
        [ D, -D, -D],  # 1
        [-D,  D, -D],  # 2
        [ D,  D, -D],  # 3
        [-D, -D,  D],  # 4
        [ D, -D,  D],  # 5
        [-D,  D,  D],  # 6
        [ D,  D,  D],  # 7
    ])


def make_icosahedron_vertices():
    """Icosahedron vertices scaled to min edge = 6.0."""
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    base_verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            base_verts.append([0.0, s1 * 1.0, s2 * phi_g])
            base_verts.append([s1 * 1.0, s2 * phi_g, 0.0])
            base_verts.append([s1 * phi_g, 0.0, s2 * 1.0])
    base_verts = np.array(base_verts)
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            dists.append(np.linalg.norm(base_verts[i] - base_verts[j]))
    scale = 6.0 / min(dists)
    return base_verts * scale


# =============================================================================
#  WORKER FUNCTION
# =============================================================================

def run_simulation(args):
    """Run a single N=128 simulation with checkpointing."""
    config_name, n_oscillons, verts, phases = args

    # Import engine inside worker (for multiprocessing spawn)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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

    print("")
    print("=" * 60)
    if resume_from is not None:
        print("  [%s] RESUMING N=%d simulation" % (tag, N_GRID))
    else:
        print("  [%s] Starting N=%d simulation (fresh)" % (tag, N_GRID))
    print("  Oscillons: %d" % n_oscillons)
    print("  Phases: %s" % list(phases))
    print("=" * 60)
    sys.stdout.flush()

    # Build initial conditions
    phi_init = np.zeros((N_GRID, N_GRID, N_GRID))
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

    # Checkpoint callback
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

    result = {
        "completed": True,
        "metadata": {
            "geometry": config_name,
            "config_name": config_name,
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": wall_time,
        },
        "parameters": {
            "phi0": phi0, "R": R, "m": m, "g4": g4, "g6": g6,
            "N_grid": N_GRID, "L": L, "dt": dt, "T_final": T_END,
            "sigma_KO": sigma, "d_edge": 6.0,
        },
        "initial_conditions": {
            "n_oscillons": n_oscillons,
            "phases": list(phases),
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

    # Atomic write
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
    print("  Phase 4b: Selection Rule Convergence Test (N=128)")
    print("  4 simulations in parallel using Pool(4)")
    print("  Checkpoints every %d steps" % CHECKPOINT_EVERY)
    print("  Output dir: %s" % os.path.abspath(OUTPUT_DIR))
    print("=" * 65)
    print("")

    # Verify N=128 baseline exists
    if not os.path.exists(BASELINE_N128_JSON):
        print("ERROR: N=128 baseline not found at %s" % BASELINE_N128_JSON)
        print("Run 08_convergence_N128.py first.")
        sys.exit(1)

    with open(BASELINE_N128_JSON) as f:
        baseline = json.load(f)
    print("N=128 baseline loaded: %d time points, E(0)=%.6f, E(500)=%.6f" % (
        len(baseline["time_series"]["times"]),
        baseline["time_series"]["E_total"][0],
        baseline["time_series"]["E_total"][-1]))
    print("")

    # Geometry definitions
    cube_verts = make_cube_vertices()
    ico_verts = make_icosahedron_vertices()

    # Phase assignments (verified from existing output files)
    # Cube all-same: [1,1,1,1,1,1,1,1], 0 cross-edges, E_bind_N64 = +73.84
    # Cube checkerboard: [-1,-1,-1,1,1,1,-1,1], 6 cross-edges, E_bind_N64 = -3.86
    # Ico ce_15: [-1,1,1,-1,-1,1,1,1,1,1,1,1], 15 cross-edges, E_bind_N64 = +3.56
    # Ico ce_20: [-1,-1,1,-1,-1,1,1,1,-1,-1,1,1], 20 cross-edges, E_bind_N64 = -50.11

    work = [
        ("cube_allsame_N128", 8, cube_verts,
         [1, 1, 1, 1, 1, 1, 1, 1]),
        ("cube_checkerboard_N128", 8, cube_verts,
         [-1, -1, -1, 1, 1, 1, -1, 1]),
        ("ico_ce15A_N128", 12, ico_verts,
         [-1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1]),
        ("ico_ce20_N128", 12, ico_verts,
         [-1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1]),
    ]

    # Check status of each config
    for name, _, _, _ in work:
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

    print("Running 4 configs with Pool(4)...")
    print("  Memory estimate: 4 x ~200 MB = ~800 MB")
    print("")
    sys.stdout.flush()

    with multiprocessing.Pool(4) as pool:
        results = pool.map(run_simulation, work)

    # Compute binding energies using N=128 baseline
    from scipy.interpolate import interp1d

    E_single_interp = interp1d(
        baseline["time_series"]["times"],
        baseline["time_series"]["E_total"],
        kind="linear", fill_value="extrapolate"
    )

    # N=64 reference values
    n64_refs = {
        "cube_allsame_N128":      {"E_bind_N64": 73.8417,  "n_osc": 8,  "f_cross": 0.000},
        "cube_checkerboard_N128": {"E_bind_N64": -3.8634,  "n_osc": 8,  "f_cross": 0.500},
        "ico_ce15A_N128":         {"E_bind_N64": 3.5570,   "n_osc": 12, "f_cross": 0.500},
        "ico_ce20_N128":          {"E_bind_N64": -50.1083,  "n_osc": 12, "f_cross": 0.667},
    }

    selection_results = []

    for result, (config_name, n_osc, _, _) in zip(results, work):
        ref = n64_refs[config_name]
        times = result["time_series"]["times"]
        E_total = result["time_series"]["E_total"]

        # Compute E_bind series
        E_bind_series = [
            E_total[i] - ref["n_osc"] * float(E_single_interp(times[i]))
            for i in range(len(times))
        ]
        E_bind_N128 = E_bind_series[-1]

        # Update saved file with binding energy
        result["time_series"]["E_bind"] = E_bind_series
        result["final_state"]["E_bind_final"] = E_bind_N128
        out_path = os.path.join(OUTPUT_DIR, "%s.json" % config_name)
        tmp_path = out_path + '.tmp'
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp_path, out_path)

        E_bind_N64 = ref["E_bind_N64"]
        sign_N64 = "positive" if E_bind_N64 > 0 else "negative"
        sign_N128 = "positive" if E_bind_N128 > 0 else "negative"
        sign_preserved = sign_N64 == sign_N128

        # Magnitude change
        avg = 0.5 * (abs(E_bind_N64) + abs(E_bind_N128))
        mag_change_pct = abs(E_bind_N128 - E_bind_N64) / avg * 100 if avg > 1e-12 else 0.0

        entry = {
            "config_name": config_name,
            "f_cross": ref["f_cross"],
            "n_oscillons": ref["n_osc"],
            "E_bind_N64": E_bind_N64,
            "E_bind_N128": E_bind_N128,
            "sign_N64": sign_N64,
            "sign_N128": sign_N128,
            "sign_preserved": sign_preserved,
            "magnitude_change_pct": mag_change_pct,
        }
        selection_results.append(entry)

    wall_total = time.perf_counter() - wall_start_total

    # Combine with Phase 4 results
    combined = {
        "cube_polarized_T1": {"E_N64": -23.95, "E_N128": -51.53, "sign_preserved": True},
        "tet_single_flip": {"E_N64": -0.041, "E_N128": -0.041, "sign_preserved": True},
    }

    # Count signs preserved (including Phase 4)
    total_preserved = sum(1 for r in selection_results if r["sign_preserved"])
    total_preserved += 2  # Phase 4 results
    total_configs = len(selection_results) + 2

    all_preserved = total_preserved == total_configs

    # Geometry-specific labels for nice printing
    labels = {
        "cube_allsame_N128": ("Cube all-same", "cube"),
        "cube_checkerboard_N128": ("Cube checkerboard", "cube"),
        "ico_ce15A_N128": ("Ico ce_15_A (f=0.500)", "icosahedron"),
        "ico_ce20_N128": ("Ico ce_20 (f=0.667)", "icosahedron"),
    }

    # Build selection_rule_convergence.json
    summary_results = []
    for r in selection_results:
        label, geom = labels[r["config_name"]]
        summary_results.append({
            "geometry": geom,
            "config": r["config_name"].replace("_N128", ""),
            "f_cross": r["f_cross"],
            "E_bind_N64": r["E_bind_N64"],
            "E_bind_N128": r["E_bind_N128"],
            "sign_N64": r["sign_N64"],
            "sign_N128": r["sign_N128"],
            "sign_preserved": r["sign_preserved"],
        })

    summary = {
        "study": "selection_rule_convergence",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "question": "Does the sign pattern of E_bind survive at N=128?",
        "results": summary_results,
        "combined_with_phase4": combined,
        "verdict": {
            "total_configs_tested": total_configs,
            "signs_preserved": "%d/%d" % (total_preserved, total_configs),
            "selection_rule_robust": "YES" if all_preserved else "NO",
            "magnitude_convergence": "NO -- magnitudes change significantly for dense clusters",
            "conclusion": (
                "Selection rule is ROBUST: all %d configs preserve sign at N=128. "
                "Qualitative stability classification is resolution-independent." % total_configs
            ) if all_preserved else (
                "Selection rule is NOT fully robust: %d/%d signs preserved. "
                "Some classifications change at higher resolution." % (total_preserved, total_configs)
            ),
        },
        "wall_clock_minutes": wall_total / 60.0,
    }

    summary_path = os.path.join(OUTPUT_DIR, "selection_rule_convergence.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved -> %s" % summary_path)

    # =========================================================================
    #  PRINT SUMMARY TABLE
    # =========================================================================
    print("")
    print("=" * 75)
    print("  SELECTION RULE CONVERGENCE TEST")
    print("=" * 75)
    print("")
    print("  %-26s %12s %13s %14s" % ("Config", "E_bind(N=64)", "E_bind(N=128)", "Sign Preserved"))
    print("  " + "-" * 71)

    # Phase 4 results first
    print("  %-26s %+12.3f %+13.3f %14s" % (
        "Tet single-flip", -0.041, -0.041, "YES (Phase 4)"))
    print("  %-26s %+12.3f %+13.3f %14s" % (
        "Cube polarized T1", -23.950, -51.530, "YES (Phase 4)"))

    # New results
    for r in selection_results:
        label, _ = labels[r["config_name"]]
        key_marker = ""
        if "checkerboard" in r["config_name"] or "ce15" in r["config_name"]:
            key_marker = " <-- KEY"
        preserved_str = "YES" if r["sign_preserved"] else "NO !!!"
        print("  %-26s %+12.3f %+13.3f %14s%s" % (
            label, r["E_bind_N64"], r["E_bind_N128"], preserved_str, key_marker))

    print("")
    print("  Signs preserved: %d / %d" % (total_preserved, total_configs))
    print("  Selection rule robust: %s" % ("YES" if all_preserved else "NO"))
    print("")
    print("  Magnitude changes:")
    for r in selection_results:
        label, _ = labels[r["config_name"]]
        print("    %-26s N64 vs N128 = %.1f%% change" % (label, r["magnitude_change_pct"]))
    print("")
    print("  Wall-clock time: %.1f min" % (wall_total / 60.0))
    print("=" * 75)


if __name__ == "__main__":
    main()
