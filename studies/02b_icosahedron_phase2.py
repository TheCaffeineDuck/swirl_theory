"""
studies/02b_icosahedron_phase2.py
==================================
Phase 2: 12-oscillon icosahedron arrangement
8 configurations targeting cross-edge counts: 0, 8, 13, 14, 15, 16, 18, 20
Min edge = 6.0 (scaled from golden-ratio base vertices)
Binding energy E_bind(t) = E_total(t) - 12 * E_single(t)

Uses engine evolve() with incremental checkpointing and resume support.
"""

import sys
import os
import json
import time
import itertools

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
N_STEPS      = int(T_END / dt)   # 10000
RECORD_EVERY = 10
PRINT_EVERY  = 1000
CHECKPOINT_EVERY = 1000

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
PHASE1_JSON = os.path.join(BASE_DIR, "outputs", "phase1", "set_B_baseline.json")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "phase2", "icosahedron")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Icosahedron vertices scaled to min edge = 6.0 ----------------------------
phi_g = (1.0 + np.sqrt(5.0)) / 2.0

# Base vertices: all even permutations of (0, +-1, +-phi_g)
base_verts = []
for s1 in [1, -1]:
    for s2 in [1, -1]:
        base_verts.append([0.0,  s1 * 1.0,  s2 * phi_g])
        base_verts.append([ s1 * 1.0,  s2 * phi_g, 0.0])
        base_verts.append([ s1 * phi_g, 0.0,  s2 * 1.0])
base_verts = np.array(base_verts)  # 12 vertices

# Scale so minimum edge = 6.0
dists = []
for i in range(12):
    for j in range(i + 1, 12):
        dists.append(np.linalg.norm(base_verts[i] - base_verts[j]))
min_edge = min(dists)
scale = 6.0 / min_edge
VERTS = base_verts * scale

# Icosahedron edges: each vertex has 5 neighbors at min edge distance
EDGE_TOL = min_edge * scale * 1.01
EDGES = []
for i in range(12):
    for j in range(i + 1, 12):
        if np.linalg.norm(VERTS[i] - VERTS[j]) < EDGE_TOL * 1.01:
            EDGES.append((i, j))
# Should be 30 edges

print("Icosahedron: %d vertices, %d edges, min_edge=%.4f (target 6.0)" % (
    len(VERTS), len(EDGES), min([np.linalg.norm(VERTS[i]-VERTS[j]) for i,j in EDGES])))
print("Vertex positions:")
for idx, v in enumerate(VERTS):
    print("  v%02d = (%.4f, %.4f, %.4f)" % (idx, v[0], v[1], v[2]))
sys.stdout.flush()

def count_cross_edges(phases):
    return sum(1 for (i, j) in EDGES if phases[i] != phases[j])

# -- Find phase configs for target cross-edge counts --------------------------
TARGET_CE = [0, 8, 13, 14, 15, 16, 18, 20]
N_MAX_EDGES = len(EDGES)  # 30

print("\nSearching 4096 configs for targets %s ..." % TARGET_CE)
sys.stdout.flush()

found = {}   # target -> phases array
for phases_int in range(4096):
    phases = np.array([(1 if (phases_int >> k) & 1 == 0 else -1) for k in range(12)])
    ce = count_cross_edges(phases)
    if ce in TARGET_CE and ce not in found:
        found[ce] = phases.copy()
    if len(found) == len(TARGET_CE):
        break

print("Found configs for cross-edge counts: %s" % sorted(found.keys()))
missing = [t for t in TARGET_CE if t not in found]
if missing:
    print("WARNING: could not find configs for targets: %s" % missing)
sys.stdout.flush()

CONFIGS = sorted(found.items())  # list of (ce_count, phases)

# -- Checkpoint helpers -------------------------------------------------------
def _checkpoint_path(ce_count):
    return os.path.join(OUTPUT_DIR, "ce_%d.checkpoint.json" % ce_count)


def _output_path(ce_count):
    return os.path.join(OUTPUT_DIR, "ce_%d.json" % ce_count)


def _make_checkpoint_callback(ce_count):
    """Return a callback that atomically writes checkpoint JSON."""
    ckpt_path = _checkpoint_path(ce_count)
    tmp_path = ckpt_path + ".tmp"

    def _callback(state_dict):
        with open(tmp_path, "w") as f:
            json.dump(state_dict, f, separators=(",", ":"))
        os.replace(tmp_path, ckpt_path)
        print("  [ce_%d] checkpoint saved (step %d, t=%.1f)" % (
            ce_count, state_dict['completed_steps'], state_dict['t']))
        sys.stdout.flush()

    return _callback


# -- Evolution routine --------------------------------------------------------
def run_config(ce_count, phases):
    n_pi = int(np.sum(phases < 0))

    print("")
    print("=" * 60)
    print("  Config: ce_%d" % ce_count)
    print("  Cross-edges: %d / %d  (%.3f)" % (ce_count, N_MAX_EDGES, ce_count / N_MAX_EDGES))
    print("  Pi-phases:   %d / 12" % n_pi)
    print("=" * 60)
    sys.stdout.flush()

    # Load baseline inside run_config (not at module level)
    with open(PHASE1_JSON) as f:
        ref = json.load(f)
    E_single_interp = interp1d(ref["t_series"], ref["E_series"],
                               kind="linear", fill_value="extrapolate")

    ev = SexticEvolver(N=N, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    # -- Check for existing checkpoint to resume from -------------------------
    ckpt_path = _checkpoint_path(ce_count)
    resume_from = None
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            resume_from = json.load(f)
        print("  Found checkpoint at step %d (t=%.1f) -- resuming" % (
            resume_from['completed_steps'], resume_from['t']))
        sys.stdout.flush()

    # -- Set up initial conditions (only used if not resuming) ----------------
    if resume_from is None:
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

    # -- Run evolution with checkpointing -------------------------------------
    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=_make_checkpoint_callback(ce_count),
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag="ce_%d" % ce_count,
    )

    # -- Compute E_bind series post-hoc ---------------------------------------
    times = state['time_series']['times']
    E_total = state['time_series']['E_total']
    max_phi_series = state['time_series']['max_amplitude']

    E_bind_series = [
        E_total[i] - 12.0 * float(E_single_interp(times[i]))
        for i in range(len(times))
    ]

    E0 = state['E0']
    E_final = E_total[-1]
    Ebind_final = E_bind_series[-1]
    amp_final = max_phi_series[-1]
    drift_final = abs(E_final - E0) / (abs(E0) + 1e-30)
    amp_ret = amp_final / phi0
    verdict = "STABLE" if Ebind_final < -1.0 else "UNSTABLE"
    wall_time = state['wall_elapsed']

    print("")
    print("  -- ce_%d complete (%.1f s) --" % (ce_count, wall_time))
    print("  E_bind(T) = %.4e  |  verdict: %s" % (Ebind_final, verdict))
    sys.stdout.flush()

    # -- Save final output ----------------------------------------------------
    per_config = {
        "config_name":          "ce_%d" % ce_count,
        "cross_edges":          ce_count,
        "n_pi":                 n_pi,
        "phases":               [int(p) for p in phases],
        "t_series":             times,
        "E_series":             E_total,
        "E_bind_series":        E_bind_series,
        "max_phi_series":       max_phi_series,
        "energy_drift_final":   drift_final,
        "amplitude_retention_final": amp_ret,
        "final_E_bind":         Ebind_final,
        "verdict":              verdict,
        "runtime_seconds":      wall_time,
        "completed":            True,
    }
    out_path = _output_path(ce_count)
    with open(out_path, "w") as f:
        json.dump(per_config, f, separators=(",", ":"))
    print("  Saved -> %s" % out_path)

    # -- Delete checkpoint on successful completion ---------------------------
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("  Checkpoint removed.")
    sys.stdout.flush()

    return {
        "config_name":          "ce_%d" % ce_count,
        "cross_edges":          ce_count,
        "cross_edge_fraction":  ce_count / N_MAX_EDGES,
        "n_pi":                 n_pi,
        "final_E_bind":         Ebind_final,
        "amplitude_retention":  amp_ret,
        "energy_drift":         drift_final,
        "verdict":              verdict,
    }

# -- Main ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\n02b_icosahedron_phase2.py  --  %d configurations" % len(CONFIGS))
    print("Output dir: %s" % os.path.abspath(OUTPUT_DIR))
    sys.stdout.flush()

    summaries = []
    for ce_count, phases in CONFIGS:
        out_path = _output_path(ce_count)
        ckpt_path = _checkpoint_path(ce_count)

        # Check if final output exists with completed=True -> skip (CACHED)
        if os.path.exists(out_path):
            with open(out_path) as _f:
                _d = json.load(_f)
            if _d.get("completed", False):
                print("  Skipping ce_%d -- already completed (CACHED)" % ce_count)
                summaries.append({
                    "config_name":         _d["config_name"],
                    "cross_edges":         _d["cross_edges"],
                    "cross_edge_fraction": _d["cross_edges"] / N_MAX_EDGES,
                    "n_pi":                _d["n_pi"],
                    "final_E_bind":        _d["final_E_bind"],
                    "amplitude_retention": _d["amplitude_retention_final"],
                    "energy_drift":        _d["energy_drift_final"],
                    "verdict":             _d["verdict"],
                })
                sys.stdout.flush()
                continue

        # Otherwise run (will auto-resume from checkpoint if one exists)
        summaries.append(run_config(ce_count, phases))

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print("\nSummary saved -> %s" % summary_path)

    # Print summary table
    print("")
    print("=" * 95)
    print("  ICOSAHEDRON SUMMARY TABLE")
    print("=" * 95)
    print("  %-12s %8s %10s %6s %14s %12s %10s %10s" % (
        "Config", "X-edges", "X-frac", "n_pi", "E_bind(T)", "Amp Ret", "Drift", "Verdict"))
    print("  " + "-" * 90)
    for s in summaries:
        print("  %-12s %8d %10.3f %6d %14.4e %12.4f %10.3e %10s" % (
            s["config_name"], s["cross_edges"], s["cross_edge_fraction"],
            s["n_pi"], s["final_E_bind"], s["amplitude_retention"],
            s["energy_drift"], s["verdict"]))
    print("=" * 95)
    print("  Phase 2 Icosahedron complete.")
