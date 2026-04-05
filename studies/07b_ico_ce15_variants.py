#!/usr/bin/env python3
"""
studies/07b_ico_ce15_variants.py
================================
Phase 7a: Additional icosahedron ce_15 variants.

Enumerates all ce_15 equivalence classes under icosahedral symmetry + global
phase flip, selects 5-7 new variants, runs simulations, and analyzes the
correlation between neighbor signature variance and binding energy.

Reference: Phase 7a task spec
"""

import os
import sys

# Prevent Numba thread oversubscription (COMPUTE_GUIDELINES.md)
os.environ['NUMBA_NUM_THREADS'] = '1'

import json
import time
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.interpolate import interp1d
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Parameters (Set B, per study plan)
# ---------------------------------------------------------------------------
N       = 64
L       = 50.0
m       = 1.0
g4      = 0.30
g6      = 0.055
dt      = 0.05
sigma   = 0.01
phi0    = 0.5
R_osc   = 2.5
T_END   = 500.0
D_EDGE  = 6.0
N_STEPS      = int(T_END / dt)   # 10000
RECORD_EVERY = 10
PRINT_EVERY  = 1000
CHECKPOINT_EVERY = 1000
N_WORKERS    = 4

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
PHASE1_JSON = os.path.join(BASE_DIR, "outputs", "phase1", "set_B_baseline.json")
OUTPUT_DIR  = os.path.join(BASE_DIR, "geometric_binding_study", "outputs",
                           "icosahedron_threshold")
FIGURE_DIR  = os.path.join(BASE_DIR, "outputs", "figures")
PAPER_DATA  = os.path.join(BASE_DIR, "outputs", "paper_v4_data.json")


# ---------------------------------------------------------------------------
# Icosahedron geometry (golden ratio construction)
# ---------------------------------------------------------------------------
def build_icosahedron(d_edge):
    """Build icosahedron vertices and edges scaled to min edge = d_edge."""
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0

    base_verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            base_verts.append([0.0,  s1 * 1.0,  s2 * phi_g])
            base_verts.append([ s1 * 1.0,  s2 * phi_g, 0.0])
            base_verts.append([ s1 * phi_g, 0.0,  s2 * 1.0])
    base_verts = np.array(base_verts)

    # Scale so minimum edge = d_edge
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            dists.append(np.linalg.norm(base_verts[i] - base_verts[j]))
    min_edge = min(dists)
    scale = d_edge / min_edge
    verts = base_verts * scale

    # Edges: vertex pairs at minimum distance
    edge_tol = d_edge * 1.02
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.linalg.norm(verts[i] - verts[j]) < edge_tol:
                edges.append((i, j))
    assert len(edges) == 30, "Expected 30 icosahedron edges, got %d" % len(edges)

    return verts, edges


def count_cross_edges(phases, edges):
    """Count edges connecting opposite-phase vertices."""
    return sum(1 for (i, j) in edges if phases[i] != phases[j])


def get_neighbor_cross_counts(phases, edges, n_verts=12):
    """For each vertex, count how many of its neighbors are opposite-phase."""
    cross_counts = [0] * n_verts
    for (i, j) in edges:
        if phases[i] != phases[j]:
            cross_counts[i] += 1
            cross_counts[j] += 1
    return cross_counts


def get_neighbor_signature(phases, edges, n_verts=12):
    """Return sorted tuple of per-vertex cross-neighbor counts."""
    return tuple(sorted(get_neighbor_cross_counts(phases, edges, n_verts)))


def neighbor_signature_variance(phases, edges, n_verts=12):
    """Variance of the per-vertex cross-neighbor counts."""
    counts = get_neighbor_cross_counts(phases, edges, n_verts)
    return float(np.var(counts))


# ---------------------------------------------------------------------------
# Icosahedral symmetry group (60 rotations as vertex permutations)
# ---------------------------------------------------------------------------
def find_icosahedral_symmetries(verts):
    """Find all 60 rotational symmetries of the icosahedron as permutations.

    For each valid triple mapping (3 non-coplanar vertices -> 3 vertices),
    solve for the rotation matrix R and verify it maps all vertices correctly.

    Returns list of permutation arrays (each length 12).
    """
    n = len(verts)
    tol = 1e-6

    # Pick 3 non-coplanar source vertices (0, 1, and one that's not colinear)
    src = verts[:3]  # vertices 0, 1, 2
    # Verify non-coplanar
    cross = np.cross(src[1] - src[0], src[2] - src[0])
    assert np.linalg.norm(cross) > 0.1, "Source vertices are coplanar"

    src_mat = src.T  # 3x3 matrix
    src_inv = np.linalg.inv(src_mat)

    perms = []
    seen = set()

    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                # Target: vertices i, j, k
                tgt = np.array([verts[i], verts[j], verts[k]])
                tgt_mat = tgt.T  # 3x3

                # Solve R @ src_mat = tgt_mat -> R = tgt_mat @ src_inv
                R = tgt_mat @ src_inv

                # Check if R is a proper rotation: R^T R = I, det(R) = +1
                if abs(np.linalg.det(R) - 1.0) > tol:
                    continue
                if np.max(np.abs(R.T @ R - np.eye(3))) > tol:
                    continue

                # Check if R maps ALL vertices to vertices
                mapped = (R @ verts.T).T  # (12, 3)
                perm = np.full(n, -1, dtype=int)
                valid = True
                for a in range(n):
                    found = False
                    for b in range(n):
                        if np.linalg.norm(mapped[a] - verts[b]) < tol * D_EDGE:
                            perm[a] = b
                            found = True
                            break
                    if not found:
                        valid = False
                        break

                if not valid:
                    continue
                if len(set(perm)) != n:
                    continue

                perm_tuple = tuple(perm)
                if perm_tuple not in seen:
                    seen.add(perm_tuple)
                    perms.append(perm)

    assert len(perms) == 60, "Expected 60 rotational symmetries, got %d" % len(perms)
    return perms


def canonical_form(phases, symmetry_perms):
    """Find canonical form of phase assignment under symmetry + global flip.

    Apply all 60 rotational symmetries and global flip (120 total),
    return lexicographically smallest representation.
    """
    best = tuple(phases)
    ph = np.array(phases)

    for perm in symmetry_perms:
        # Apply permutation
        permuted = ph[perm]
        t = tuple(permuted)
        if t < best:
            best = t
        # Apply permutation + global flip
        flipped = tuple(-permuted)
        if flipped < best:
            best = flipped

    return best


# ---------------------------------------------------------------------------
# Step 1-2: Enumerate ce_15 equivalence classes and select variants
# ---------------------------------------------------------------------------
def enumerate_ce15_classes(edges, symmetry_perms):
    """Find all ce_15 phase assignments, cluster by icosahedral symmetry."""
    print("\n--- Step 1: Enumerating all 4096 phase configs ---")

    # Enumerate all 2^12 = 4096 phase assignments
    ce15_configs = []
    for bits in range(4096):
        phases = np.array([(1 if (bits >> k) & 1 == 0 else -1) for k in range(12)])
        ce = count_cross_edges(phases, edges)
        if ce == 15:
            ce15_configs.append(phases)

    print("  Found %d configs with exactly 15 cross-edges" % len(ce15_configs))

    # Cluster by symmetry equivalence
    classes = {}  # canonical_form -> list of phase arrays
    for phases in ce15_configs:
        canon = canonical_form(phases, symmetry_perms)
        if canon not in classes:
            classes[canon] = []
        classes[canon].append(phases)

    print("  Distinct equivalence classes: %d" % len(classes))

    # Build class info
    class_info = []
    for idx, (canon, members) in enumerate(sorted(classes.items()), 1):
        rep = np.array(canon)
        n_pi = int(np.sum(rep < 0))
        sig = get_neighbor_signature(rep, edges)
        var = neighbor_signature_variance(rep, edges)
        class_info.append({
            'class_id': idx,
            'canonical': canon,
            'representative': rep,
            'n_pi': n_pi,
            'signature': sig,
            'variance': round(var, 4),
            'count': len(members),
        })

    # Print table
    print("\n=== ce_15 EQUIVALENCE CLASSES ===")
    print("  Total configs with exactly 15 cross-edges: %d" % len(ce15_configs))
    print("  Distinct equivalence classes: %d" % len(class_info))
    print("")
    print("  Class | n_pi | Sig. Variance | Count | Representative Phases")
    print("  ------|------|---------------|-------|----------------------")
    for ci in sorted(class_info, key=lambda x: (x['n_pi'], x['variance'])):
        phase_str = str([int(x) for x in ci['representative']])
        print("  %5d | %4d | %13.4f | %5d | %s" % (
            ci['class_id'], ci['n_pi'], ci['variance'], ci['count'], phase_str))

    return ce15_configs, class_info


def identify_existing_classes(class_info, edges, symmetry_perms):
    """Identify which equivalence classes correspond to existing A/B/C configs."""
    existing = {
        'ce_15_A': np.array([-1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1]),
        'ce_15_B': np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1]),
        'ce_15_C': np.array([-1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1]),
    }

    existing_class_ids = {}
    for name, phases in existing.items():
        canon = canonical_form(phases, symmetry_perms)
        # Find matching class by canonical form
        for ci in class_info:
            if ci['canonical'] == canon:
                existing_class_ids[name] = ci['class_id']
                break
        if name not in existing_class_ids:
            print("  WARNING: Could not identify class for %s" % name)

    return existing_class_ids


def select_new_variants(class_info, existing_class_ids, edges, symmetry_perms):
    """Select new ce_15 variants from all remaining equivalence classes."""
    print("\n--- Step 2: Selecting new variants ---")

    used_class_ids = set(existing_class_ids.values())

    # Available classes (not already used)
    available = [ci for ci in class_info if ci['class_id'] not in used_class_ids]

    # Sort by variance to get good spread
    available.sort(key=lambda x: (x['variance'], x['class_id']))

    if len(available) == 0:
        print("  No additional classes available!")
        return []

    # Take ALL available classes (there are only 3 remaining out of 6 total)
    selected = available[:]

    # Print existing
    print("\n=== SELECTED NEW ce_15 VARIANTS ===")
    print("\n  Existing (already run):")
    for name, cid in sorted(existing_class_ids.items()):
        ci = next(c for c in class_info if c['class_id'] == cid)
        print("    %s: class %d, n_pi=%d, var=%.4f, sig=%s" % (
            name, cid, ci['n_pi'], ci['variance'], ci['signature']))

    print("\n  New selections:")
    # Assign labels D, E, F, ...
    # Use representative phases with n_pi <= 6 (flip if needed for lower n_pi)
    labels = []
    for i, ci in enumerate(selected):
        label = "ce_15_%s" % chr(ord('D') + i)
        labels.append(label)
        # Use the representative directly (canonical form)
        rep = ci['representative']
        n_pi_rep = int(np.sum(rep < 0))
        # If n_pi > 6, flip to get the equivalent config with n_pi <= 6
        if n_pi_rep > 6:
            rep = -rep
            n_pi_rep = 12 - n_pi_rep
        print("    %s: class %d, n_pi=%d, var=%.4f, sig=%s, phases=%s" % (
            label, ci['class_id'], n_pi_rep, ci['variance'],
            get_neighbor_signature(rep, edges),
            str([int(x) for x in rep])))

    # Build config dicts
    new_configs = []
    for i, ci in enumerate(selected):
        label = labels[i]
        rep = ci['representative'].copy()
        n_pi_rep = int(np.sum(rep < 0))
        # Flip to get lower n_pi if needed
        if n_pi_rep > 6:
            rep = -rep
            n_pi_rep = 12 - n_pi_rep
        new_configs.append({
            'config_name': label,
            'phases': [int(x) for x in rep],
            'n_pi': n_pi_rep,
            'class_id': ci['class_id'],
            'variance': ci['variance'],
            'signature': get_neighbor_signature(rep, edges),
        })

    print("\n  Total: %d existing + %d new = %d (of %d total equivalence classes)" % (
        len(existing_class_ids), len(new_configs),
        len(existing_class_ids) + len(new_configs), len(class_info)))

    return new_configs


# ---------------------------------------------------------------------------
# Step 3: Simulation worker
# ---------------------------------------------------------------------------
def run_single_config(config):
    """Run one ce_15 simulation. Called by Pool workers."""
    import numpy as _np
    from scipy.interpolate import interp1d as _interp1d
    from engine.evolver import SexticEvolver, serialize_field

    config_name = config['config_name']
    phases = _np.array(config['phases'])
    verts = _np.array(config['vertex_positions'])
    n_pi = config['n_pi']
    output_path = config['output_path']
    checkpoint_path = output_path.replace('.json', '.checkpoint.json')

    # Check cache
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
            if existing.get('completed', False):
                print("  [%s] CACHED -- skipping" % config_name)
                sys.stdout.flush()
                return {
                    "config_name": config_name,
                    "n_pi": n_pi,
                    "phases": [int(p) for p in phases],
                    "E_bind_final": existing["final_state"]["E_bind_final"],
                    "verdict": existing["final_state"]["verdict"],
                    "amp_retention": existing["final_state"]["amplitude_retention"],
                    "runtime_seconds": existing["metadata"]["runtime_seconds"],
                }
        except (json.JSONDecodeError, KeyError):
            pass

    # Load baseline
    with open(config['baseline_path']) as f:
        ref = json.load(f)
    E_single_interp = _interp1d(ref["t_series"], ref["E_series"],
                                kind="linear", fill_value="extrapolate")

    # Verify cross-edge count
    edges = config['edges']
    ce = sum(1 for (i, j) in edges if phases[i] != phases[j])
    assert ce == 15, "%s has %d cross-edges, expected 15!" % (config_name, ce)
    print("  [%s] VERIFIED: %d cross-edges. Starting n_pi=%d" % (config_name, ce, n_pi))
    sys.stdout.flush()

    # Check for checkpoint
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

    ev = SexticEvolver(N=N, L=L, m=m, g4=g4, g6=g6, dissipation_sigma=sigma)

    # Build initial conditions
    phi_init = _np.zeros((N, N, N))
    for idx in range(12):
        A = phases[idx]
        pos = verts[idx]
        dx_ = ev.X - pos[0]
        dy_ = ev.Y - pos[1]
        dz_ = ev.Z - pos[2]
        r2 = dx_**2 + dy_**2 + dz_**2
        phi_init += A * phi0 * _np.exp(-r2 / (2.0 * R_osc**2))

    phi_dot_init = _np.zeros_like(phi_init)
    ev.set_initial_conditions(phi_init, phi_dot_init)

    if resume_from is None:
        E0 = ev.compute_energy()
        amp0 = float(_np.max(_np.abs(phi_init)))
        print("  [%s] E(0) = %.6e  max|phi|(0) = %.6f" % (config_name, E0, amp0))
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
        tag=config_name,
    )

    wall_time = state['wall_elapsed']
    E0 = state['E0']
    max_phi0 = state['max_phi0']
    times = state['time_series']['times']
    E_series = state['time_series']['E_total']
    max_phi_series = state['time_series']['max_amplitude']

    # Compute E_bind series
    Ebind_series = []
    econs_series = []
    for i in range(len(times)):
        e_ref = float(E_single_interp(times[i]))
        e_bind = E_series[i] - 12.0 * e_ref
        drift = abs(E_series[i] - E0) / (abs(E0) + 1e-30)
        Ebind_series.append(e_bind)
        econs_series.append(drift)

    E_final     = E_series[-1]
    Ebind_final = Ebind_series[-1]
    amp_final   = max_phi_series[-1]
    drift_final = econs_series[-1]
    amp_ret     = amp_final / max_phi0 if max_phi0 > 0 else 0.0
    verdict     = "STABLE" if Ebind_final < -1.0 else ("MARGINAL" if Ebind_final < 0 else "UNSTABLE")

    print("  [%s] DONE (%.1f s) -- E_bind(T)=%.4f  verdict=%s" % (
        config_name, wall_time, Ebind_final, verdict))
    sys.stdout.flush()

    # Build output JSON
    result = {
        "completed": True,
        "metadata": {
            "geometry": "icosahedron",
            "config_name": config_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": round(wall_time, 2),
            "study": "phase7a_ce15_variants"
        },
        "parameters": {
            "phi0": phi0, "R": R_osc, "m": m, "g4": g4, "g6": g6,
            "N_grid": N, "L": L, "dt": dt, "T_final": T_END,
            "sigma_KO": sigma, "d_edge": D_EDGE
        },
        "initial_conditions": {
            "n_oscillons": 12,
            "phases": [int(p) for p in phases],
            "vertex_positions": verts.tolist(),
            "n_pi": n_pi,
            "cross_edges": 15,
            "f_cross": 0.500,
        },
        "time_series": {
            "times": times,
            "E_total": E_series,
            "E_bind": Ebind_series,
            "max_amplitude": max_phi_series,
            "energy_conservation_pct": econs_series
        },
        "final_state": {
            "E_bind_final": round(Ebind_final, 6),
            "E_total_final": round(E_final, 6),
            "max_amplitude_final": round(amp_final, 6),
            "amplitude_retention": round(amp_ret, 6),
            "energy_drift_pct": round(drift_final, 8),
            "verdict": verdict
        }
    }

    # Atomic write
    tmp_path = output_path + '.tmp'
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, output_path)
    print("  [%s] Saved -> %s" % (config_name, output_path))
    sys.stdout.flush()

    # Delete checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return {
        "config_name": config_name,
        "n_pi": n_pi,
        "phases": [int(p) for p in phases],
        "E_bind_final": Ebind_final,
        "verdict": verdict,
        "amp_retention": amp_ret,
        "runtime_seconds": round(wall_time, 2)
    }


# ---------------------------------------------------------------------------
# Step 4: Correlation analysis
# ---------------------------------------------------------------------------
def analyze_correlation(all_data):
    """Compute correlation between signature variance and E_bind."""
    print("\n--- Step 4: Correlation analysis ---")

    variances = np.array([d['variance'] for d in all_data])
    ebinds = np.array([d['E_bind'] for d in all_data])
    npis = np.array([d['n_pi'] for d in all_data])

    # Pearson
    r_pearson, p_pearson = stats.pearsonr(variances, ebinds)
    # Spearman
    r_spearman, p_spearman = stats.spearmanr(variances, ebinds)
    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(variances, ebinds)
    r_squared = r_value**2

    # Partial correlation controlling for n_pi
    # Method: regress both variance and E_bind on n_pi, correlate residuals
    if len(set(npis)) > 1:
        slope_v, int_v, _, _, _ = stats.linregress(npis, variances)
        slope_e, int_e, _, _, _ = stats.linregress(npis, ebinds)
        resid_v = variances - (slope_v * npis + int_v)
        resid_e = ebinds - (slope_e * npis + int_e)
        if np.std(resid_v) > 1e-10 and np.std(resid_e) > 1e-10:
            partial_r, partial_p = stats.pearsonr(resid_v, resid_e)
        else:
            partial_r, partial_p = 0.0, 1.0
    else:
        partial_r = r_pearson
        partial_p = p_pearson

    # Check if n_pi correlates with variance
    if len(set(npis)) > 1:
        r_npi_var, p_npi_var = stats.pearsonr(npis, variances)
    else:
        r_npi_var, p_npi_var = 0.0, 1.0

    # Determine strength
    abs_r = abs(r_pearson)
    if abs_r > 0.7:
        strength = "Strong"
    elif abs_r > 0.4:
        strength = "Moderate"
    elif abs_r > 0.2:
        strength = "Weak"
    else:
        strength = "No"

    print("\n=== CORRELATION ANALYSIS ===")
    print("")
    print("  Pearson r (variance vs E_bind): %.4f (p = %.4f)" % (r_pearson, p_pearson))
    print("  Spearman rho: %.4f (p = %.4f)" % (r_spearman, p_spearman))
    print("  Linear fit: E_bind = %.3f * variance + %.3f (R^2 = %.4f)" % (
        slope, intercept, r_squared))
    print("")
    print("  Controlling for n_pi:")
    print("    n_pi vs variance correlation: r = %.4f (p = %.4f)" % (r_npi_var, p_npi_var))
    print("    Partial correlation (variance vs E_bind | n_pi): %.4f" % partial_r)
    print("")
    print("  Conclusion: %s correlation between signature variance" % strength)
    print("              and binding energy at f_cross = 0.500")

    return {
        'pearson_r': round(r_pearson, 4),
        'pearson_p': round(p_pearson, 4),
        'spearman_rho': round(r_spearman, 4),
        'spearman_p': round(p_spearman, 4),
        'linear_fit': {
            'slope': round(slope, 4),
            'intercept': round(intercept, 4),
            'r_squared': round(r_squared, 4),
        },
        'partial_correlation_controlling_npi': round(partial_r, 4),
        'strength': strength,
    }


# ---------------------------------------------------------------------------
# Step 5: Update paper_v4_data.json
# ---------------------------------------------------------------------------
def update_paper_data(all_data, correlation):
    """Add ce_15 variants and correlation to paper_v4_data.json."""
    print("\n--- Step 5: Updating paper_v4_data.json ---")

    with open(PAPER_DATA) as f:
        paper = json.load(f)

    # Build ce_15_variants list
    variants = []
    for d in sorted(all_data, key=lambda x: x['name']):
        variants.append({
            "name": d['name'],
            "n_pi": d['n_pi'],
            "cross_edges": 15,
            "f_cross": 0.500,
            "E_bind": d['E_bind'],
            "neighbor_signature_variance": d['variance'],
            "verdict": d['verdict'],
        })

    paper['geometries']['icosahedron']['ce_15_variants'] = variants
    paper['geometries']['icosahedron']['ce_15_correlation'] = correlation

    # Atomic write
    tmp_path = PAPER_DATA + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(paper, f, indent=2)
    os.replace(tmp_path, PAPER_DATA)
    print("  Updated %s" % PAPER_DATA)


# ---------------------------------------------------------------------------
# Step 6: Generate scatter plot
# ---------------------------------------------------------------------------
def generate_scatter_plot(all_data, correlation):
    """Create E_bind vs neighbor signature variance scatter plot."""
    print("\n--- Step 6: Generating scatter plot ---")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib import cm

    os.makedirs(FIGURE_DIR, exist_ok=True)

    variances = [d['variance'] for d in all_data]
    ebinds = [d['E_bind'] for d in all_data]
    npis = [d['n_pi'] for d in all_data]
    names = [d['name'] for d in all_data]
    existing_names = {'ce_15_A', 'ce_15_B', 'ce_15_C'}

    fig, ax = plt.subplots(figsize=(7, 5))

    # Colormap by n_pi
    npi_min, npi_max = min(npis), max(npis)
    if npi_min == npi_max:
        npi_max = npi_min + 1
    norm = Normalize(vmin=npi_min - 0.5, vmax=npi_max + 0.5)
    cmap = cm.viridis

    # Plot new points
    for i, d in enumerate(all_data):
        color = cmap(norm(d['n_pi']))
        if d['name'] in existing_names:
            ax.scatter(d['variance'], d['E_bind'], c=[color], s=120,
                       edgecolors='red', linewidths=2, zorder=5)
            ax.annotate(d['name'].split('_')[-1], (d['variance'], d['E_bind']),
                        textcoords="offset points", xytext=(6, 6), fontsize=9,
                        fontweight='bold', color='red')
        else:
            ax.scatter(d['variance'], d['E_bind'], c=[color], s=80,
                       edgecolors='black', linewidths=1, zorder=4)
            ax.annotate(d['name'].split('_')[-1], (d['variance'], d['E_bind']),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)

    # Linear fit line
    fit = correlation['linear_fit']
    var_range = np.linspace(min(variances) - 0.1, max(variances) + 0.1, 100)
    ebind_fit = fit['slope'] * var_range + fit['intercept']
    ax.plot(var_range, ebind_fit, 'k--', alpha=0.5, linewidth=1.5,
            label='Linear fit (R$^2$=%.3f)' % fit['r_squared'])

    # E_bind = 0 line
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='n_pi')
    cbar.set_ticks(range(npi_min, npi_max + 1))

    ax.set_xlabel('Neighbor Signature Variance', fontsize=12)
    ax.set_ylabel('E_bind', fontsize=12)
    ax.set_title('Icosahedron ce_15: Binding Energy vs Phase Uniformity', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, loc='upper left')

    # Annotate R^2
    ax.text(0.98, 0.02, 'Pearson r = %.3f\nR$^2$ = %.3f' % (
        correlation['pearson_r'], fit['r_squared']),
        transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    png_path = os.path.join(FIGURE_DIR, 'ce15_variance_scatter.png')
    pdf_path = os.path.join(FIGURE_DIR, 'ce15_variance_scatter.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print("  Saved %s" % png_path)
    print("  Saved %s" % pdf_path)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def cleanup_checkpoints(output_dir):
    """Remove checkpoint and temp files."""
    import glob as _glob
    checkpoints = _glob.glob(os.path.join(str(output_dir), '*.checkpoint*'))
    temps = _glob.glob(os.path.join(str(output_dir), '*.tmp'))
    total_freed = 0
    for f in checkpoints + temps:
        try:
            size = os.path.getsize(f)
            os.remove(f)
            total_freed += size
        except OSError:
            pass
    if checkpoints or temps:
        print("  Cleanup: removed %d checkpoints, %d temp files, freed %.1f MB" % (
            len(checkpoints), len(temps), total_freed / 1e6))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print("=" * 70)
    print("  PHASE 7a: ICOSAHEDRON ce_15 VARIANTS")
    print("  Additional variants to test variance-binding correlation")
    print("=" * 70)

    # Build geometry
    verts, edges = build_icosahedron(D_EDGE)
    actual_min = min(np.linalg.norm(verts[i] - verts[j]) for i, j in edges)
    print("\nIcosahedron: 12 vertices, %d edges, min_edge=%.4f" % (len(edges), actual_min))

    # Find symmetry group
    print("\nFinding icosahedral symmetry group...")
    t0 = time.perf_counter()
    symmetry_perms = find_icosahedral_symmetries(verts)
    print("  Found %d rotational symmetries (%.1f s)" % (
        len(symmetry_perms), time.perf_counter() - t0))

    # Step 1: Enumerate equivalence classes
    ce15_configs, class_info = enumerate_ce15_classes(edges, symmetry_perms)

    # Identify existing A/B/C
    existing_class_ids = identify_existing_classes(class_info, edges, symmetry_perms)

    # Step 2: Select new variants
    new_configs = select_new_variants(class_info, existing_class_ids, edges, symmetry_perms)

    if len(new_configs) == 0:
        print("\nNo new variants to run. Exiting.")
        return

    # Step 3: Run simulations
    print("\n" + "=" * 70)
    print("  Running %d new configs on %d workers..." % (len(new_configs), N_WORKERS))
    print("  Output dir: %s" % os.path.abspath(OUTPUT_DIR))
    print("=" * 70)

    # Add shared fields for workers
    for c in new_configs:
        c['vertex_positions'] = verts.tolist()
        c['baseline_path'] = os.path.abspath(PHASE1_JSON)
        c['output_path'] = os.path.join(OUTPUT_DIR, "ico_%s.json" % c['config_name'])
        c['edges'] = edges

    # Verify all have ce=15
    print("\nPre-flight verification:")
    for c in new_configs:
        ce = count_cross_edges(np.array(c['phases']), edges)
        print("  %s: phases=%s  CE=%d  n_pi=%d  var=%.4f  -- %s" % (
            c['config_name'], c['phases'], ce, c['n_pi'], c['variance'],
            "OK" if ce == 15 else "FAIL"))
        assert ce == 15, "%s has %d cross-edges!" % (c['config_name'], ce)

    total_start = time.perf_counter()

    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single_config, new_configs)

    total_wall = time.perf_counter() - total_start
    print("\nAll simulations complete. Wall time: %.1f min" % (total_wall / 60.0))

    # Cleanup checkpoint files
    cleanup_checkpoints(OUTPUT_DIR)

    # Step 4: Compile full dataset and analyze
    print("\n" + "=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    # Load existing A/B/C data
    existing_data = []
    for name in ['ce_15_A', 'ce_15_B', 'ce_15_C']:
        fpath = os.path.join(OUTPUT_DIR, "ico_%s.json" % name)
        with open(fpath) as f:
            d = json.load(f)
        phases_arr = np.array(d['initial_conditions']['phases'])
        var = neighbor_signature_variance(phases_arr, edges)
        existing_data.append({
            'name': name,
            'n_pi': d['initial_conditions']['n_pi'],
            'cross_edges': 15,
            'E_bind': d['final_state']['E_bind_final'],
            'variance': round(var, 4),
            'verdict': d['final_state']['verdict'],
            'amp_retention': d['final_state']['amplitude_retention'],
        })

    # Build new data from results
    new_data = []
    for r in results:
        phases_arr = np.array(r['phases'])
        var = neighbor_signature_variance(phases_arr, edges)
        new_data.append({
            'name': r['config_name'],
            'n_pi': r['n_pi'],
            'cross_edges': 15,
            'E_bind': r['E_bind_final'],
            'variance': round(var, 4),
            'verdict': r['verdict'],
            'amp_retention': r['amp_retention'],
        })

    all_data = existing_data + new_data

    # Print full dataset
    print("\n=== ce_15 COMPLETE DATASET ===")
    print("")
    print("  Config  | n_pi | Cross-Edges | Sig. Variance | E_bind   | Verdict")
    print("  --------|------|-------------|---------------|----------|--------")
    for d in sorted(all_data, key=lambda x: x['variance']):
        print("  %-7s | %4d | %11d | %13.4f | %+8.4f | %s" % (
            d['name'].split('_')[-1], d['n_pi'], d['cross_edges'],
            d['variance'], d['E_bind'], d['verdict']))

    # Correlation analysis
    correlation = analyze_correlation(all_data)

    # Step 5: Update paper data
    update_paper_data(all_data, correlation)

    # Step 6: Generate figure
    generate_scatter_plot(all_data, correlation)

    # Step 7: Summary
    variances = [d['variance'] for d in all_data]
    ebinds = [d['E_bind'] for d in all_data]

    direction = "Higher variance -> more positive E_bind" if correlation['pearson_r'] > 0 else "Higher variance -> more negative E_bind"
    if abs(correlation['pearson_r']) < 0.3:
        direction = "No clear trend"

    if abs(correlation['pearson_r']) > 0.6:
        implication = "Strong correlation supports secondary predictor claim"
    elif abs(correlation['pearson_r']) > 0.3:
        implication = "Moderate correlation -- signature variance is a meaningful secondary predictor"
    else:
        implication = "Weak correlation -- signature variance is suggestive but not definitive"

    print("\n" + "=" * 70)
    print("  PHASE 7a COMPLETE")
    print("=" * 70)
    print("")
    print("  New configs run: %d" % len(new_data))
    print("  Total ce_15 dataset: %d configs (3 existing + %d new)" % (
        len(all_data), len(new_data)))
    print("  Signature variance range: [%.4f, %.4f]" % (min(variances), max(variances)))
    print("  E_bind range: [%.4f, %.4f]" % (min(ebinds), max(ebinds)))
    print("")
    print("  Correlation:")
    print("    Pearson r = %.4f" % correlation['pearson_r'])
    print("    R^2 = %.4f" % correlation['linear_fit']['r_squared'])
    print("    Direction: %s" % direction)
    print("")
    print("  Paper implication:")
    print("    %s" % implication)
    print("")
    print("  Files saved:")
    for d in new_data:
        print("    outputs/icosahedron_threshold/ico_%s.json" % d['name'])
    print("    outputs/paper_v4_data.json (updated)")
    print("    outputs/figures/ce15_variance_scatter.png")
    print("    outputs/figures/ce15_variance_scatter.pdf")
    print("")
    print("  Wall-clock time: %.1f min" % (total_wall / 60.0))
    print("=" * 70)


if __name__ == '__main__':
    main()
