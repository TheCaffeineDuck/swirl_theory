#!/usr/bin/env python3
"""
studies/phase12_self_organization.py
=====================================
Phase 12: Self-Organization Ensemble

Determine whether icosahedral oscillon clusters with random continuous initial
phases spontaneously self-organize toward configurations satisfying the
selection rule (Sigma_edges cos(Delta_phi_e) < 0).

50-seed ensemble at T=1000 (~177 oscillation periods) with phase tracking
every 10 time units. Parallelized across 4 P-cores.
"""

import os
import sys
import json
import time
import multiprocessing as mp

# Prevent Numba thread oversubscription when using multiprocessing
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np

# Ensure engine is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
N_WORKERS = 4

# Physics parameters (Set B)
N_GRID  = 64
L       = 50.0
M       = 1.0
G4      = 0.30
G6      = 0.055
DT      = 0.05
SIGMA   = 0.01
PHI0    = 0.5
R_GAUSS = 2.5
OMEGA_EFF = 1.113

# Evolution
T_FINAL    = 1000.0
N_STEPS    = int(T_FINAL / DT)       # 20000
MEAS_DT    = 10.0                     # measurement interval in time units
RECORD_EVERY = int(MEAS_DT / DT)     # 200 steps between measurements
PRINT_EVERY  = 2000
CHECKPOINT_EVERY = 2000               # ~100 time units between checkpoints

N_SEEDS = 50
MIN_AMP_GUARD = 0.01 * PHI0          # skip phase measurement below this

# Directories
BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "phase12")

# ---------------------------------------------------------------------------
#  Icosahedron geometry (same as 02b)
# ---------------------------------------------------------------------------
phi_g = (1.0 + np.sqrt(5.0)) / 2.0

base_verts = []
for s1 in [1, -1]:
    for s2 in [1, -1]:
        base_verts.append([0.0,  s1 * 1.0,  s2 * phi_g])
        base_verts.append([ s1 * 1.0,  s2 * phi_g, 0.0])
        base_verts.append([ s1 * phi_g, 0.0,  s2 * 1.0])
base_verts = np.array(base_verts)

dists = []
for i in range(12):
    for j in range(i + 1, 12):
        dists.append(np.linalg.norm(base_verts[i] - base_verts[j]))
min_edge = min(dists)
scale = 6.0 / min_edge
VERTS = base_verts * scale

EDGE_TOL = min_edge * scale * 1.01
EDGES = []
for i in range(12):
    for j in range(i + 1, 12):
        if np.linalg.norm(VERTS[i] - VERTS[j]) < EDGE_TOL * 1.01:
            EDGES.append((i, j))

assert len(VERTS) == 12, "Expected 12 vertices, got %d" % len(VERTS)
assert len(EDGES) == 30, "Expected 30 edges, got %d" % len(EDGES)


# ---------------------------------------------------------------------------
#  Helper: grid indices nearest to a vertex position
# ---------------------------------------------------------------------------
def _nearest_grid_idx(ev, pos):
    """Return (ix, iy, iz) indices of the grid point nearest to pos."""
    dx = ev.L / ev.N
    # Grid runs from -L/2 to L/2-dx
    ix = int(round((pos[0] + ev.L / 2.0) / dx)) % ev.N
    iy = int(round((pos[1] + ev.L / 2.0) / dx)) % ev.N
    iz = int(round((pos[2] + ev.L / 2.0) / dx)) % ev.N
    return ix, iy, iz


# ---------------------------------------------------------------------------
#  Phase measurement diagnostic function
# ---------------------------------------------------------------------------
def make_phase_diagnostic(verts, edges, omega_eff, min_amp):
    """Return a diagnostic function compatible with evolver.evolve(extra_diagnostic_fn=...)."""
    # Pre-compute grid indices (set once per evolver instance)
    grid_indices = None

    def diagnostic(ev):
        nonlocal grid_indices
        if grid_indices is None:
            grid_indices = [_nearest_grid_idx(ev, v) for v in verts]

        phases = []
        for idx, (ix, iy, iz) in enumerate(grid_indices):
            phi_val = ev.phi[ix, iy, iz]
            phi_dot_val = ev.phi_dot[ix, iy, iz]
            amp = np.sqrt((omega_eff * phi_val)**2 + phi_dot_val**2)
            if amp < min_amp:
                # Low amplitude -- carry forward NaN, will be handled later
                phases.append(float('nan'))
            else:
                theta = np.arctan2(-phi_dot_val, omega_eff * phi_val)
                phases.append(float(theta))

        # Order parameter: S = (1/30) * sum_edges cos(theta_i - theta_j)
        s_val = 0.0
        n_valid = 0
        for (i, j) in edges:
            if np.isnan(phases[i]) or np.isnan(phases[j]):
                continue
            s_val += np.cos(phases[i] - phases[j])
            n_valid += 1
        if n_valid > 0:
            s_val /= n_valid
        else:
            s_val = float('nan')

        # Circular standard deviation of phases
        valid_phases = [p for p in phases if not np.isnan(p)]
        if len(valid_phases) > 1:
            c = np.mean(np.cos(valid_phases))
            s = np.mean(np.sin(valid_phases))
            R_bar = np.sqrt(c**2 + s**2)
            circ_std = np.sqrt(-2.0 * np.log(max(R_bar, 1e-15)))
        else:
            circ_std = float('nan')

        return {
            'phases': phases,
            'S': float(s_val),
            'f_eff': float((1.0 - s_val) / 2.0) if not np.isnan(s_val) else float('nan'),
            'circ_std': float(circ_std),
        }

    return diagnostic


# ---------------------------------------------------------------------------
#  Checkpoint helpers
# ---------------------------------------------------------------------------
def _checkpoint_path(label):
    return os.path.join(OUTPUT_DIR, "%s.checkpoint.json" % label)


def _output_path(label):
    return os.path.join(OUTPUT_DIR, "%s.json" % label)


def _make_checkpoint_callback(label):
    ckpt_path = _checkpoint_path(label)
    tmp_path = ckpt_path + ".tmp"

    def _callback(state_dict):
        with open(tmp_path, "w") as f:
            json.dump(state_dict, f, separators=(",", ":"))
        os.replace(tmp_path, ckpt_path)

    return _callback


# ---------------------------------------------------------------------------
#  Baseline: single isolated oscillon at theta=0, T=1000
# ---------------------------------------------------------------------------
def run_baseline():
    """Run one isolated oscillon and return time-matched E_single series."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from engine.evolver import SexticEvolver

    label = "baseline_T1000"
    out_path = _output_path(label)

    # Cache check
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        if data.get("completed", False):
            print("CACHED: baseline")
            sys.stdout.flush()
            return data

    print("Running baseline (single oscillon, T=%.0f) ..." % T_FINAL)
    sys.stdout.flush()

    ev = SexticEvolver(N=N_GRID, L=L, m=M, g4=G4, g6=G6, dissipation_sigma=SIGMA)

    # Single oscillon at center, theta=0
    phi_init = PHI0 * np.exp(-(ev.X**2 + ev.Y**2 + ev.Z**2) / (2.0 * R_GAUSS**2))
    phi_dot_init = np.zeros_like(phi_init)  # cos(0)=1, sin(0)=0
    ev.set_initial_conditions(phi_init, phi_dot_init)

    # Check for resume
    ckpt_path = _checkpoint_path(label)
    resume_from = None
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            resume_from = json.load(f)
        print("  RESUMING baseline from step %d" % resume_from['completed_steps'])
        sys.stdout.flush()

    state = ev.evolve(
        dt=DT,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=_make_checkpoint_callback(label),
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag="baseline",
    )

    result = {
        "completed": True,
        "label": "baseline_T1000",
        "times": state['time_series']['times'],
        "E_single": state['time_series']['E_total'],
        "max_amplitude": state['time_series']['max_amplitude'],
        "E0": state['E0'],
        "energy_drift_pct": abs(state['time_series']['E_total'][-1] - state['E0'])
                            / (abs(state['E0']) + 1e-30),
        "wall_time_seconds": state['wall_elapsed'],
    }

    # Atomic write
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out_path)

    # Cleanup checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print("Baseline complete: E0=%.6e, drift=%.2e, wall=%.0fs" % (
        result['E0'], result['energy_drift_pct'], result['wall_time_seconds']))
    sys.stdout.flush()
    return result


# ---------------------------------------------------------------------------
#  Single seed worker
# ---------------------------------------------------------------------------
def run_seed(seed):
    """Run one ensemble member. Called by Pool workers."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from engine.evolver import SexticEvolver
    from scipy.interpolate import interp1d

    label = "seed_%02d" % seed
    out_path = _output_path(label)

    # Cache check
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                data = json.load(f)
            if data.get("completed", False):
                print("CACHED: %s" % label)
                sys.stdout.flush()
                return data
        except (json.JSONDecodeError, OSError):
            pass

    wall_start = time.perf_counter()

    # Load baseline for binding energy
    baseline_path = _output_path("baseline_T1000")
    with open(baseline_path) as f:
        baseline = json.load(f)
    E_single_interp = interp1d(baseline['times'], baseline['E_single'],
                               kind='linear', fill_value='extrapolate')

    # Generate random phases
    rng = np.random.default_rng(seed)
    initial_phases = rng.uniform(0, 2 * np.pi, size=12).tolist()

    # Initialize evolver
    ev = SexticEvolver(N=N_GRID, L=L, m=M, g4=G4, g6=G6, dissipation_sigma=SIGMA)

    phi_init = np.zeros((N_GRID, N_GRID, N_GRID))
    phi_dot_init = np.zeros((N_GRID, N_GRID, N_GRID))

    for idx in range(12):
        theta_i = initial_phases[idx]
        pos = VERTS[idx]
        dx_ = ev.X - pos[0]
        dy_ = ev.Y - pos[1]
        dz_ = ev.Z - pos[2]
        r2 = dx_**2 + dy_**2 + dz_**2
        envelope = np.exp(-r2 / (2.0 * R_GAUSS**2))
        phi_init += PHI0 * np.cos(theta_i) * envelope
        phi_dot_init += (-OMEGA_EFF * PHI0 * np.sin(theta_i)) * envelope

    ev.set_initial_conditions(phi_init, phi_dot_init)

    # Phase diagnostic function
    diag_fn = make_phase_diagnostic(VERTS, EDGES, OMEGA_EFF, MIN_AMP_GUARD)

    # Check for resume
    ckpt_path = _checkpoint_path(label)
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  RESUMING %s from step %d" % (label, resume_from['completed_steps']))
            sys.stdout.flush()
        except (json.JSONDecodeError, OSError):
            resume_from = None

    # Evolve
    state = ev.evolve(
        dt=DT,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=_make_checkpoint_callback(label),
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=label,
        extra_diagnostic_fn=diag_fn,
    )

    # Extract time series
    times = state['time_series']['times']
    E_total = state['time_series']['E_total']
    extras = state.get('extra_diagnostics', [])

    order_parameter = [d['S'] for d in extras]
    f_eff = [d['f_eff'] for d in extras]
    phases_series = [d['phases'] for d in extras]

    # Binding energy: E_total(t) - 12 * E_single(t)
    binding_energy = [
        E_total[i] - 12.0 * float(E_single_interp(times[i]))
        for i in range(len(times))
    ]

    E0 = state['E0']
    energy_drift_pct = abs(E_total[-1] - E0) / (abs(E0) + 1e-30)
    wall_time = time.perf_counter() - wall_start

    # Compute initial S from the diagnostic at t=0
    initial_S = order_parameter[0] if order_parameter else float('nan')
    initial_f_eff = f_eff[0] if f_eff else float('nan')

    result = {
        "completed": True,
        "seed": seed,
        "initial_phases": initial_phases,
        "initial_S": initial_S,
        "initial_f_eff": initial_f_eff,
        "times": times,
        "order_parameter": order_parameter,
        "f_eff": f_eff,
        "binding_energy": binding_energy,
        "phases": phases_series,
        "total_energy": E_total,
        "energy_drift_pct": energy_drift_pct,
        "wall_time_seconds": wall_time,
    }

    # Flag high energy drift
    if energy_drift_pct > 0.0005:
        print("  WARNING: %s energy drift %.4f%% exceeds 0.05%%" % (
            label, energy_drift_pct * 100))
        sys.stdout.flush()

    # Atomic write
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    os.replace(tmp, out_path)

    # Cleanup checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    S_final = order_parameter[-1] if order_parameter else float('nan')
    E_bind_final = binding_energy[-1] if binding_energy else float('nan')
    print("Seed %02d complete: S_final=%.3f, E_bind_final=%.2f (%.0fs)" % (
        seed, S_final, E_bind_final, wall_time))
    sys.stdout.flush()

    return result


# ---------------------------------------------------------------------------
#  Ensemble summary
# ---------------------------------------------------------------------------
def build_summary():
    """Load all 50 seed results and build ensemble_summary.json."""
    print("\n" + "=" * 70)
    print("  ENSEMBLE SUMMARY")
    print("=" * 70)
    sys.stdout.flush()

    results = []
    for seed in range(N_SEEDS):
        path = _output_path("seed_%02d" % seed)
        if not os.path.exists(path):
            print("  WARNING: missing seed_%02d" % seed)
            continue
        with open(path) as f:
            data = json.load(f)
        if not data.get("completed", False):
            print("  WARNING: seed_%02d incomplete" % seed)
            continue
        results.append(data)

    if len(results) == 0:
        print("  No completed seeds found!")
        return

    S_initials = [r['initial_S'] for r in results if not np.isnan(r['initial_S'])]
    S_finals = [r['order_parameter'][-1] for r in results
                if r['order_parameter'] and not np.isnan(r['order_parameter'][-1])]
    delta_S = [r['order_parameter'][-1] - r['initial_S'] for r in results
               if r['order_parameter'] and not np.isnan(r['order_parameter'][-1])
               and not np.isnan(r['initial_S'])]
    E_bind_finals = [r['binding_energy'][-1] for r in results if r['binding_energy']]

    mean_S_initial = float(np.mean(S_initials)) if S_initials else float('nan')
    mean_S_final = float(np.mean(S_finals)) if S_finals else float('nan')
    std_S_final = float(np.std(S_finals)) if S_finals else float('nan')
    mean_delta_S = float(np.mean(delta_S)) if delta_S else float('nan')
    std_delta_S = float(np.std(delta_S)) if delta_S else float('nan')

    n_decreased = sum(1 for ds in delta_S if ds < 0)
    n_bound = sum(1 for eb in E_bind_finals if eb < 0)
    mean_E_bind_final = float(np.mean(E_bind_finals)) if E_bind_finals else float('nan')

    # Significance test: is mean_delta_S < 0 at >2 sigma?
    n_valid = len(delta_S)
    if n_valid > 1 and std_delta_S > 0:
        sem = std_delta_S / np.sqrt(n_valid)
        z_score = mean_delta_S / sem
        self_org = mean_delta_S < 0 and z_score < -2.0
    else:
        z_score = float('nan')
        self_org = False

    # Total wall time
    total_wall = sum(r.get('wall_time_seconds', 0) for r in results)

    summary = {
        "n_seeds": len(results),
        "total_wall_time": total_wall,
        "self_organization_detected": bool(self_org),
        "mean_S_initial": mean_S_initial,
        "mean_S_final": mean_S_final,
        "std_S_final": std_S_final,
        "mean_delta_S": mean_delta_S,
        "std_delta_S": std_delta_S,
        "z_score": float(z_score),
        "n_seeds_S_decreased": n_decreased,
        "n_seeds_bound_at_final": n_bound,
        "mean_E_bind_final": mean_E_bind_final,
    }

    summary_path = _output_path("ensemble_summary")
    tmp = summary_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, summary_path)

    # Print summary table
    print("")
    print("  Seeds completed:        %d / %d" % (len(results), N_SEEDS))
    print("  Total wall time:        %.1f min" % (total_wall / 60.0))
    print("")
    print("  mean S(t=0):            %.4f" % mean_S_initial)
    print("  mean S(t=1000):         %.4f" % mean_S_final)
    print("  mean delta_S:           %.4f  (std=%.4f, z=%.2f)" % (
        mean_delta_S, std_delta_S, z_score))
    print("  Seeds S decreased:      %d / %d (%.0f%%)" % (
        n_decreased, n_valid, 100.0 * n_decreased / max(n_valid, 1)))
    print("  Seeds bound (E<0):      %d / %d (%.0f%%)" % (
        n_bound, len(E_bind_finals), 100.0 * n_bound / max(len(E_bind_finals), 1)))
    print("  mean E_bind(t=1000):    %.4f" % mean_E_bind_final)
    print("")
    print("  Self-organization:      %s" % ("YES" if self_org else "NO"))
    print("=" * 70)

    # Per-seed table
    print("")
    print("  %-8s %10s %10s %10s %12s %10s" % (
        "Seed", "S_init", "S_final", "delta_S", "E_bind_fin", "drift%"))
    print("  " + "-" * 62)
    for r in results:
        s_i = r['initial_S']
        s_f = r['order_parameter'][-1] if r['order_parameter'] else float('nan')
        ds = s_f - s_i if not (np.isnan(s_f) or np.isnan(s_i)) else float('nan')
        eb = r['binding_energy'][-1] if r['binding_energy'] else float('nan')
        dr = r.get('energy_drift_pct', float('nan'))
        print("  seed_%02d %10.4f %10.4f %10.4f %12.4f %10.4f" % (
            r['seed'], s_i, s_f, ds, eb, dr * 100))
    print("")
    sys.stdout.flush()

    return summary


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Phase 12: Self-Organization Ensemble")
    print("  Geometry:    Icosahedron (12 vertices, 30 edges, d=6.0)")
    print("  Parameters:  Set B (m=1.0, g4=0.30, g6=0.055)")
    print("  Grid:        N=%d, L=%.1f, dt=%.3f, sigma=%.3f" % (N_GRID, L, DT, SIGMA))
    print("  Evolution:   T=%.0f (~%d oscillation periods)" % (
        T_FINAL, int(T_FINAL * OMEGA_EFF / (2 * np.pi))))
    print("  Ensemble:    %d seeds, measurements every dt=%.0f" % (N_SEEDS, MEAS_DT))
    print("  Workers:     %d P-cores" % N_WORKERS)
    print("  Output:      %s" % os.path.abspath(OUTPUT_DIR))
    print("")
    sys.stdout.flush()

    # Stage 1: Baseline
    t0 = time.perf_counter()
    baseline = run_baseline()
    print("")

    # Stage 2: Ensemble (parallel)
    print("Starting %d-seed ensemble on %d workers ..." % (N_SEEDS, N_WORKERS))
    sys.stdout.flush()

    seeds = list(range(N_SEEDS))
    with mp.Pool(N_WORKERS) as pool:
        pool.map(run_seed, seeds)

    total_time = time.perf_counter() - t0
    print("\nAll seeds complete. Total wall time: %.1f min" % (total_time / 60.0))
    sys.stdout.flush()

    # Stage 3: Summary
    build_summary()

    # Cleanup
    import glob as gl
    checkpoints = gl.glob(os.path.join(OUTPUT_DIR, "*.checkpoint*"))
    temps = gl.glob(os.path.join(OUTPUT_DIR, "*.tmp"))
    for f in checkpoints + temps:
        try:
            os.remove(f)
        except OSError:
            pass
    if checkpoints or temps:
        print("Cleanup: removed %d checkpoints, %d temp files" % (
            len(checkpoints), len(temps)))

    print("\nPhase 12 complete.")


if __name__ == '__main__':
    main()
