"""
studies/04_new_geometries.py
============================
Phase 4: New Geometry Universality Study

Tests the 50% cross-edge stability threshold on two new Platonic solids:
  - Octahedron (N=6 vertices, 12 edges, NOT bipartite -- contains triangles)
  - Tetrahedron (N=4 vertices, 6 edges, complete K4, non-bipartite)

Uses multiprocessing.Pool(4) for parallel config execution.
Loads Phase 1 baseline for time-matched binding energy.
Prints final comparison table across all 4 geometries (cube, icosahedron,
octahedron, tetrahedron).

Supports incremental checkpointing via engine's evolve() method.
"""

import sys
import os
import json
import multiprocessing

import numpy as np
from scipy.interpolate import interp1d

os.environ["NUMBA_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.evolver import SexticEvolver

# -- Parameters (Set B -- matches Phase 1 baseline and Phase 2 runs) ----------
N_GRID = 64
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
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "phase4_new_geometries")

# -- Phase 2 summaries (for final comparison table) ---------------------------
CUBE_SUMMARY = os.path.join(BASE_DIR, "outputs", "phase2", "cube", "summary.json")
ICO_SUMMARY  = os.path.join(BASE_DIR, "outputs", "phase2", "icosahedron", "summary.json")


# =============================================================================
#  GEOMETRY DEFINITIONS
# =============================================================================

def make_octahedron():
    """Octahedron: 6 vertices, 12 edges, NOT bipartite (contains triangles)."""
    d = 6.0
    verts = np.array([
        [ d/2,   0,   0],  # 0  Set A
        [   0, d/2,   0],  # 1  Set A
        [   0,   0, d/2],  # 2  Set A
        [-d/2,   0,   0],  # 3  Set B
        [   0,-d/2,   0],  # 4  Set B
        [   0,   0,-d/2],  # 5  Set B
    ])
    # Edges: every vertex in Set A connects to every vertex in Set B (4 neighbors each)
    # Plus edges within sets? No -- in a regular octahedron each vertex connects to
    # exactly the 4 non-antipodal vertices. Let's compute explicitly.
    edges = []
    edge_len = d / np.sqrt(2)  # distance between adjacent octahedron vertices
    tol = edge_len * 1.01
    for i in range(6):
        for j in range(i + 1, 6):
            dist = np.linalg.norm(verts[i] - verts[j])
            if dist < tol:
                edges.append((i, j))
    return verts, edges


def make_tetrahedron():
    """Regular tetrahedron: 4 vertices, 6 edges (complete K4), edge length = 6.0."""
    # Standard tetrahedron inscribed in cube of side 2, then scaled
    base = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float)
    # Current edge length = 2*sqrt(2)
    current_edge = 2.0 * np.sqrt(2.0)
    scale = 6.0 / current_edge
    verts = base * scale
    # K4: all pairs are edges
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    return verts, edges


def count_cross_edges(phases, edges):
    return sum(1 for (i, j) in edges if phases[i] != phases[j])


# =============================================================================
#  OCTAHEDRON CONFIGS
# =============================================================================

OCT_VERTS, OCT_EDGES = make_octahedron()
N_OCT_EDGES = len(OCT_EDGES)

# Antipodal pairs: (0,3), (1,4), (2,5). Each vertex connects to 4 (all except antipodal).
OCT_CONFIGS = [
    # all_same: all +1, 0 cross-edges -> UNSTABLE
    ("all_same",     [+1, +1, +1, +1, +1, +1]),
    # single_flip: one vertex flipped, 4 cross-edges (33%) -> UNSTABLE
    ("single_flip",  [-1, +1, +1, +1, +1, +1]),
    # balanced: flip two adjacent vertices (e.g. v0, v1) -> 6 cross-edges (50%) -> measure
    ("balanced",     [-1, -1, +1, +1, +1, +1]),
    # polarized: flip one antipodal pair (v0, v3)
    # -> 8 cross-edges (67%, max achievable for octahedron) -> STABLE
    ("polarized",    [-1, +1, +1, -1, +1, +1]),
]


# =============================================================================
#  TETRAHEDRON CONFIGS
# =============================================================================

TET_VERTS, TET_EDGES = make_tetrahedron()
N_TET_EDGES = len(TET_EDGES)

TET_CONFIGS = [
    # all_same: 0 cross-edges -> UNSTABLE
    ("all_same",          [+1, +1, +1, +1]),
    # single_flip (n_pi=1): vertex 0 flipped, connects to 3 others -> 3 cross (50%)
    ("single_flip",       [-1, +1, +1, +1]),
    # two_flip_adjacent (n_pi=2): vertices 0,1 flipped -> 4 cross-edges (67%)
    ("two_flip_adjacent", [-1, -1, +1, +1]),
    # two_flip_opposite: same as adjacent in K4 (all pairs connected) -> 4 cross (67%)
    # But we label it differently for completeness: vertices 0,3 flipped
    ("two_flip_opposite", [-1, +1, +1, -1]),
]


# =============================================================================
#  EVOLUTION ROUTINE
# =============================================================================

def run_config(args):
    """Run a single configuration. Designed for multiprocessing.Pool."""
    geom_name, config_name, phases, verts, edges, n_total_edges, n_oscillons = args
    phases = np.array(phases, dtype=float)
    n_cross = count_cross_edges(phases, edges)
    n_pi = int(np.sum(phases < 0))

    tag = "%s/%s" % (geom_name, config_name)

    # -- Output paths --
    geom_dir = os.path.join(OUTPUT_DIR, geom_name)
    os.makedirs(geom_dir, exist_ok=True)
    out_path = os.path.join(geom_dir, "%s.json" % config_name)
    ckpt_path = os.path.join(geom_dir, "%s.checkpoint.json" % config_name)

    # -- 1. Check if final output already exists with completed=True -> skip --
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if existing.get("completed", False):
                print("")
                print("  [%s] CACHED -- final output exists, skipping" % tag)
                sys.stdout.flush()
                return {
                    "geometry":             existing["geometry"],
                    "config_name":          existing["config_name"],
                    "cross_edges":          existing["cross_edges"],
                    "cross_edge_fraction":  existing["cross_edge_fraction"],
                    "n_pi":                 existing["n_pi"],
                    "final_E_bind":         existing["final_E_bind"],
                    "amplitude_retention":  existing["amplitude_retention_final"],
                    "energy_drift":         existing["energy_drift_final"],
                    "verdict":              existing["verdict"],
                }
        except (json.JSONDecodeError, KeyError):
            pass  # corrupted file, re-run

    print("")
    print("=" * 60)
    print("  [%s]" % tag)
    print("  Cross-edges: %d / %d  (%.3f)" % (n_cross, n_total_edges,
                                                n_cross / n_total_edges))
    print("  Pi-phases:   %d / %d" % (n_pi, n_oscillons))
    print("=" * 60)
    sys.stdout.flush()

    # -- Load baseline inside worker (avoid pickling interp1d on macOS spawn) --
    with open(PHASE1_JSON) as f:
        ref = json.load(f)
    E_single_interp = interp1d(ref["t_series"], ref["E_series"],
                               kind="linear", fill_value="extrapolate")

    ev = SexticEvolver(N=N_GRID, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    # -- Build initial conditions --
    phi_init = np.zeros((N_GRID, N_GRID, N_GRID))
    for idx in range(n_oscillons):
        pos = verts[idx]
        dx_ = ev.X - pos[0]
        dy_ = ev.Y - pos[1]
        dz_ = ev.Z - pos[2]
        r2 = dx_**2 + dy_**2 + dz_**2
        phi_init += phases[idx] * phi0 * np.exp(-r2 / (2.0 * R**2))

    phi_dot_init = np.zeros_like(phi_init)
    ev.set_initial_conditions(phi_init, phi_dot_init)

    E0 = ev.compute_energy()
    print("  [%s] E(0) = %.6e  max|phi|(0) = %.6f" % (
        tag, E0, float(np.max(np.abs(phi_init)))))
    sys.stdout.flush()

    # -- 2. Check for checkpoint -> resume --
    resume_from = None
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                resume_from = json.load(f)
            print("  [%s] Found checkpoint at step %d, resuming..." % (
                tag, resume_from.get("completed_steps", 0)))
            sys.stdout.flush()
        except (json.JSONDecodeError, KeyError):
            print("  [%s] Corrupted checkpoint, starting fresh" % tag)
            sys.stdout.flush()
            resume_from = None

    # -- 3. Checkpoint callback: atomic write via .tmp + os.replace --
    def checkpoint_callback(state_dict):
        tmp_path = ckpt_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state_dict, f, separators=(",", ":"))
        os.replace(tmp_path, ckpt_path)
        print("  [%s] checkpoint saved (step %d)" % (
            tag, state_dict["completed_steps"]))
        sys.stdout.flush()

    # -- Run evolve() --
    state = ev.evolve(
        dt=dt,
        n_steps=N_STEPS,
        record_every=RECORD_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_callback=checkpoint_callback,
        resume_from=resume_from,
        print_every=PRINT_EVERY,
        tag=tag,
    )

    wall_time = state["wall_elapsed"]
    times = state["time_series"]["times"]
    E_total = state["time_series"]["E_total"]
    max_amp = state["time_series"]["max_amplitude"]

    # -- 4. Compute E_bind post-hoc --
    Ebind_series = []
    for i in range(len(times)):
        e_ref = float(E_single_interp(times[i]))
        e_bind = E_total[i] - n_oscillons * e_ref
        Ebind_series.append(e_bind)

    E_final     = E_total[-1]
    Ebind_final = Ebind_series[-1]
    amp_final   = max_amp[-1]
    drift_final = abs(E_final - E0) / (abs(E0) + 1e-30)
    amp_ret     = amp_final / phi0
    verdict     = "STABLE" if Ebind_final < -1.0 else "UNSTABLE"

    print("")
    print("  [%s] complete (%.1f s)  E_bind(T)=%.4e  verdict=%s" % (
        tag, wall_time, Ebind_final, verdict))
    sys.stdout.flush()

    # -- 5. Save final result JSON (existing schema + completed=True) --
    per_config = {
        "geometry":             geom_name,
        "config_name":          config_name,
        "cross_edges":          n_cross,
        "cross_edge_fraction":  n_cross / n_total_edges,
        "n_pi":                 n_pi,
        "phases":               [int(p) for p in phases],
        "t_series":             times,
        "E_series":             E_total,
        "E_bind_series":        Ebind_series,
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

    # -- Delete checkpoint after successful completion --
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("  [%s] checkpoint deleted" % tag)

    sys.stdout.flush()

    return {
        "geometry":             geom_name,
        "config_name":          config_name,
        "cross_edges":          n_cross,
        "cross_edge_fraction":  n_cross / n_total_edges,
        "n_pi":                 n_pi,
        "final_E_bind":         Ebind_final,
        "amplitude_retention":  amp_ret,
        "energy_drift":         drift_final,
        "verdict":              verdict,
    }


# =============================================================================
#  REPORT GENERATION
# =============================================================================

def write_summary_json(summaries, geom_name):
    """Save geometry summary JSON."""
    out_path = os.path.join(OUTPUT_DIR, "%s_summary.json" % geom_name)
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print("  Saved -> %s" % out_path)
    return out_path


def load_phase2_summaries():
    """Load cube and icosahedron Phase 2 summaries for comparison table."""
    phase2 = {}
    for label, path in [("cube", CUBE_SUMMARY), ("icosahedron", ICO_SUMMARY)]:
        if os.path.exists(path):
            with open(path) as f:
                phase2[label] = json.load(f)
        else:
            print("  WARNING: %s not found, skipping in comparison" % path)
    return phase2


def find_threshold(data, total_edges):
    """Find last unstable and first stable cross-edge fractions."""
    sorted_data = sorted(data, key=lambda x: x["cross_edge_fraction"])
    last_unstable = None
    first_stable = None
    for cfg in sorted_data:
        if cfg["verdict"] == "UNSTABLE":
            last_unstable = cfg
        elif cfg["verdict"] == "STABLE" and first_stable is None:
            first_stable = cfg
    return {
        "total_edges": total_edges,
        "last_unstable_frac": last_unstable["cross_edge_fraction"] if last_unstable else None,
        "last_unstable_name": last_unstable["config_name"] if last_unstable else None,
        "first_stable_frac": first_stable["cross_edge_fraction"] if first_stable else None,
        "first_stable_name": first_stable["config_name"] if first_stable else None,
    }


def write_report(oct_results, tet_results, phase2):
    """Write new_geometries_report.md."""
    lines = []
    lines.append("# Phase 4: New Geometry Universality Study")
    lines.append("")
    lines.append("## Simulation Parameters")
    lines.append("- m=%.1f, g4=%.2f, g6=%.3f" % (m, g4, g6))
    lines.append("- phi0=%.1f, R=%.1f" % (phi0, R))
    lines.append("- Grid: N=%d, L=%.1f, dt=%.2f, T=%.0f" % (N_GRID, L, dt, T_END))
    lines.append("- RK4 + spectral Laplacian + KO dissipation (sigma=%.2f)" % sigma)
    lines.append("")

    # Octahedron section
    lines.append("## Octahedron (6 vertices, 12 edges, bipartite)")
    lines.append("")
    lines.append("| Config | Cross-Edges | X-Frac | E_bind(T) | Drift | Verdict |")
    lines.append("|--------|-------------|--------|-----------|-------|---------|")
    for s in sorted(oct_results, key=lambda x: x["cross_edge_fraction"]):
        lines.append("| %-15s | %5d | %.3f | %+12.4e | %.2e | %-8s |" % (
            s["config_name"], s["cross_edges"], s["cross_edge_fraction"],
            s["final_E_bind"], s["energy_drift"], s["verdict"]))
    lines.append("")

    oct_thresh = find_threshold(oct_results, N_OCT_EDGES)
    if oct_thresh["last_unstable_frac"] is not None and oct_thresh["first_stable_frac"] is not None:
        lines.append("**Threshold:** between %.3f (%s) and %.3f (%s)" % (
            oct_thresh["last_unstable_frac"], oct_thresh["last_unstable_name"],
            oct_thresh["first_stable_frac"], oct_thresh["first_stable_name"]))
    else:
        lines.append("**Threshold:** could not be determined from configs tested")
    lines.append("")

    # Tetrahedron section
    lines.append("## Tetrahedron (4 vertices, 6 edges, complete K4, non-bipartite)")
    lines.append("")
    lines.append("| Config | Cross-Edges | X-Frac | E_bind(T) | Drift | Verdict |")
    lines.append("|--------|-------------|--------|-----------|-------|---------|")
    for s in sorted(tet_results, key=lambda x: x["cross_edge_fraction"]):
        lines.append("| %-20s | %5d | %.3f | %+12.4e | %.2e | %-8s |" % (
            s["config_name"], s["cross_edges"], s["cross_edge_fraction"],
            s["final_E_bind"], s["energy_drift"], s["verdict"]))
    lines.append("")

    tet_thresh = find_threshold(tet_results, N_TET_EDGES)
    if tet_thresh["last_unstable_frac"] is not None and tet_thresh["first_stable_frac"] is not None:
        lines.append("**Threshold:** between %.3f (%s) and %.3f (%s)" % (
            tet_thresh["last_unstable_frac"], tet_thresh["last_unstable_name"],
            tet_thresh["first_stable_frac"], tet_thresh["first_stable_name"]))
    else:
        lines.append("**Threshold:** could not be determined from configs tested")
    lines.append("")

    # Cross-geometry comparison table
    lines.append("## Cross-Geometry 50%% Threshold Comparison")
    lines.append("")
    lines.append("| Geometry | Type | V | E | Bipartite | Threshold Range | Notes |")
    lines.append("|----------|------|---|---|-----------|-----------------|-------|")

    all_geoms = []

    # Add Phase 2 geometries
    if "cube" in phase2:
        cube_thresh = find_threshold(phase2["cube"], 12)
        lu = "%.3f" % cube_thresh["last_unstable_frac"] if cube_thresh["last_unstable_frac"] is not None else "N/A"
        fs = "%.3f" % cube_thresh["first_stable_frac"] if cube_thresh["first_stable_frac"] is not None else "N/A"
        lines.append("| Cube | Q3 (3-regular) | 8 | 12 | Yes | %s - %s | Phase 2 |" % (lu, fs))
        all_geoms.append(("Cube", cube_thresh))

    if "icosahedron" in phase2:
        ico_thresh = find_threshold(phase2["icosahedron"], 30)
        lu = "%.3f" % ico_thresh["last_unstable_frac"] if ico_thresh["last_unstable_frac"] is not None else "N/A"
        fs = "%.3f" % ico_thresh["first_stable_frac"] if ico_thresh["first_stable_frac"] is not None else "N/A"
        lines.append("| Icosahedron | Regular (5-regular) | 12 | 30 | No | %s - %s | Phase 2 |" % (lu, fs))
        all_geoms.append(("Icosahedron", ico_thresh))

    # Octahedron
    lu = "%.3f" % oct_thresh["last_unstable_frac"] if oct_thresh["last_unstable_frac"] is not None else "N/A"
    fs = "%.3f" % oct_thresh["first_stable_frac"] if oct_thresh["first_stable_frac"] is not None else "N/A"
    lines.append("| Octahedron | Regular (4-regular) | 6 | 12 | No | %s - %s | Phase 4 |" % (lu, fs))
    all_geoms.append(("Octahedron", oct_thresh))

    # Tetrahedron
    lu = "%.3f" % tet_thresh["last_unstable_frac"] if tet_thresh["last_unstable_frac"] is not None else "N/A"
    fs = "%.3f" % tet_thresh["first_stable_frac"] if tet_thresh["first_stable_frac"] is not None else "N/A"
    lines.append("| Tetrahedron | K4 (3-regular) | 4 | 6 | No | %s - %s | Phase 4 |" % (lu, fs))
    all_geoms.append(("Tetrahedron", tet_thresh))

    lines.append("")

    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. The 50%% cross-edge threshold has been tested across 4 Platonic-solid")
    lines.append("   geometries with varying vertex counts, edge counts, and graph properties.")
    lines.append("")
    lines.append("2. Both bipartite (cube, octahedron) and non-bipartite (icosahedron,")
    lines.append("   tetrahedron/K4) graphs are included.")
    lines.append("")
    lines.append("3. Results support/refine the universal 50%% cross-edge stability threshold.")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by studies/04_new_geometries.py*")

    report_path = os.path.join(OUTPUT_DIR, "new_geometries_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Saved report -> %s" % report_path)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  Phase 4: New Geometry Universality Study")
    print("  Octahedron (%d configs) + Tetrahedron (%d configs)" % (
        len(OCT_CONFIGS), len(TET_CONFIGS)))
    print("  Output dir: %s" % os.path.abspath(OUTPUT_DIR))
    print("=" * 65)
    print("")

    # Validate geometries
    print("Octahedron: %d vertices, %d edges" % (len(OCT_VERTS), N_OCT_EDGES))
    for i, v in enumerate(OCT_VERTS):
        print("  v%d = (%.2f, %.2f, %.2f)" % (i, v[0], v[1], v[2]))
    print("  Edges: %s" % OCT_EDGES)
    print("")

    print("Tetrahedron: %d vertices, %d edges" % (len(TET_VERTS), N_TET_EDGES))
    for i, v in enumerate(TET_VERTS):
        print("  v%d = (%.4f, %.4f, %.4f)" % (i, v[0], v[1], v[2]))
    print("  Edges: %s" % TET_EDGES)
    print("")

    # Pre-check cross-edge counts
    print("Configuration cross-edge counts:")
    for name, phases in OCT_CONFIGS:
        ce = count_cross_edges(phases, OCT_EDGES)
        print("  octahedron/%-20s  %2d / %2d  (%.3f)" % (name, ce, N_OCT_EDGES, ce / N_OCT_EDGES))
    for name, phases in TET_CONFIGS:
        ce = count_cross_edges(phases, TET_EDGES)
        print("  tetrahedron/%-20s %2d / %2d  (%.3f)" % (name, ce, N_TET_EDGES, ce / N_TET_EDGES))
    print("")
    sys.stdout.flush()

    # Build work items
    work = []
    for name, phases in OCT_CONFIGS:
        work.append(("octahedron", name, phases, OCT_VERTS, OCT_EDGES,
                      N_OCT_EDGES, len(OCT_VERTS)))
    for name, phases in TET_CONFIGS:
        work.append(("tetrahedron", name, phases, TET_VERTS, TET_EDGES,
                      N_TET_EDGES, len(TET_VERTS)))

    # Run with multiprocessing
    print("Running %d configs with Pool(4)..." % len(work))
    sys.stdout.flush()

    with multiprocessing.Pool(4) as pool:
        all_results = pool.map(run_config, work)

    # Split results by geometry
    oct_results = [r for r in all_results if r["geometry"] == "octahedron"]
    tet_results = [r for r in all_results if r["geometry"] == "tetrahedron"]

    # Save summaries
    print("")
    print("Saving summaries...")
    write_summary_json(oct_results, "octahedron")
    write_summary_json(tet_results, "tetrahedron")

    # Load Phase 2 data for comparison
    phase2 = load_phase2_summaries()

    # Write report
    print("")
    print("Writing report...")
    write_report(oct_results, tet_results, phase2)

    # Print summary tables
    print("")
    print("=" * 95)
    print("  OCTAHEDRON RESULTS (%d vertices, %d edges)" % (len(OCT_VERTS), N_OCT_EDGES))
    print("=" * 95)
    print("  %-20s %8s %10s %14s %10s %10s" % (
        "Config", "X-edges", "X-frac", "E_bind(T)", "Drift", "Verdict"))
    print("  " + "-" * 78)
    for s in sorted(oct_results, key=lambda x: x["cross_edge_fraction"]):
        print("  %-20s %8d %10.3f %14.4e %10.3e %10s" % (
            s["config_name"], s["cross_edges"], s["cross_edge_fraction"],
            s["final_E_bind"], s["energy_drift"], s["verdict"]))

    print("")
    print("=" * 95)
    print("  TETRAHEDRON RESULTS (%d vertices, %d edges)" % (len(TET_VERTS), N_TET_EDGES))
    print("=" * 95)
    print("  %-20s %8s %10s %14s %10s %10s" % (
        "Config", "X-edges", "X-frac", "E_bind(T)", "Drift", "Verdict"))
    print("  " + "-" * 78)
    for s in sorted(tet_results, key=lambda x: x["cross_edge_fraction"]):
        print("  %-20s %8d %10.3f %14.4e %10.3e %10s" % (
            s["config_name"], s["cross_edges"], s["cross_edge_fraction"],
            s["final_E_bind"], s["energy_drift"], s["verdict"]))

    # Cross-geometry comparison table
    print("")
    print("=" * 110)
    print("  50%% CROSS-EDGE THRESHOLD -- ALL 4 GEOMETRIES")
    print("=" * 110)
    print("  %-14s %6s %6s %10s  %-30s  %s" % (
        "Geometry", "V", "E", "Bipartite", "Threshold (unstable -> stable)", "Source"))
    print("  " + "-" * 104)

    if "cube" in phase2:
        ct = find_threshold(phase2["cube"], 12)
        lu = "%.3f" % ct["last_unstable_frac"] if ct["last_unstable_frac"] is not None else "N/A"
        fs = "%.3f" % ct["first_stable_frac"] if ct["first_stable_frac"] is not None else "N/A"
        print("  %-14s %6d %6d %10s  %s -> %s" % ("Cube", 8, 12, "Yes", lu, fs) +
              "                       Phase 2")

    if "icosahedron" in phase2:
        it = find_threshold(phase2["icosahedron"], 30)
        lu = "%.3f" % it["last_unstable_frac"] if it["last_unstable_frac"] is not None else "N/A"
        fs = "%.3f" % it["first_stable_frac"] if it["first_stable_frac"] is not None else "N/A"
        print("  %-14s %6d %6d %10s  %s -> %s" % ("Icosahedron", 12, 30, "No", lu, fs) +
              "                       Phase 2")

    ot = find_threshold(oct_results, N_OCT_EDGES)
    lu = "%.3f" % ot["last_unstable_frac"] if ot["last_unstable_frac"] is not None else "N/A"
    fs = "%.3f" % ot["first_stable_frac"] if ot["first_stable_frac"] is not None else "N/A"
    print("  %-14s %6d %6d %10s  %s -> %s" % ("Octahedron", 6, 12, "No", lu, fs) +
          "                       Phase 4")

    tt = find_threshold(tet_results, N_TET_EDGES)
    lu = "%.3f" % tt["last_unstable_frac"] if tt["last_unstable_frac"] is not None else "N/A"
    fs = "%.3f" % tt["first_stable_frac"] if tt["first_stable_frac"] is not None else "N/A"
    print("  %-14s %6d %6d %10s  %s -> %s" % ("Tetrahedron", 4, 6, "No", lu, fs) +
          "                       Phase 4")

    print("=" * 110)
    print("  Phase 4 complete. Output in: %s" % os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
