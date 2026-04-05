#!/usr/bin/env python3
"""
studies/phase7bc_edge_length_and_extended.py
============================================
Phase 7b: Cube at d=7.5 (5 configs, T=500)
Phase 7c: Extended evolution to T=2000 (cube polarized T1 + ico ce_20)

Uses multiprocessing.Pool. NUMBA_NUM_THREADS=1.
Checkpointing via engine/checkpoint.py.
"""

import os
import sys
import json
import time
import multiprocessing as mp

os.environ['NUMBA_NUM_THREADS'] = '1'

# Paths
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, BASE_DIR)

PHASE1_JSON = os.path.join(BASE_DIR, "outputs", "phase1", "set_B_baseline.json")
OUTPUT_DIR_7B = os.path.join(BASE_DIR, "outputs", "edge_length_d75")
OUTPUT_DIR_7C = os.path.join(BASE_DIR, "outputs", "extended_evolution")

N_WORKERS = 4

# ============================================================
# Common parameters (Set B)
# ============================================================
PARAMS = dict(N=64, L=50.0, m=1.0, g4=0.30, g6=0.055, sigma_KO=0.01)
PHI0 = 0.5
R_GAUSS = 2.5
DT = 0.05
RECORD_EVERY = 10

# ============================================================
# Cube graph definition
# ============================================================
# Vertex ordering (same as studies/02a_cube_phase2.py):
# 0=(-D,-D,-D), 1=(+D,-D,-D), 2=(-D,+D,-D), 3=(+D,+D,-D),
# 4=(-D,-D,+D), 5=(+D,-D,+D), 6=(-D,+D,+D), 7=(+D,+D,+D)

def cube_vertices(d):
    """Return 8 cube vertex positions at edge length d."""
    D = d / 2.0
    return [
        [-D, -D, -D],  # 0
        [+D, -D, -D],  # 1
        [-D, +D, -D],  # 2
        [+D, +D, -D],  # 3
        [-D, -D, +D],  # 4
        [+D, -D, +D],  # 5
        [-D, +D, +D],  # 6
        [+D, +D, +D],  # 7
    ]

# Cube edges: pairs differing in exactly one coordinate
CUBE_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7),
]
assert len(CUBE_EDGES) == 12


def count_cross_edges_cube(phases):
    """Count cross-edges for cube."""
    return sum(1 for i, j in CUBE_EDGES if phases[i] != phases[j])


# ============================================================
# Icosahedron graph definition (from run_analysis.py)
# ============================================================
# Edges from 02b_icosahedron_phase2.py vertex ordering
# (golden ratio construction: even permutations of (0, +-1, +-phi))
ICO_EDGES = [
    (0, 1), (0, 2), (0, 6), (0, 7), (0, 8),
    (1, 2), (1, 3), (1, 5), (1, 7),
    (2, 4), (2, 5), (2, 6),
    (3, 5), (3, 7), (3, 9), (3, 11),
    (4, 5), (4, 6), (4, 9), (4, 10),
    (5, 9),
    (6, 8), (6, 10),
    (7, 8), (7, 11),
    (8, 10), (8, 11),
    (9, 10), (9, 11),
    (10, 11),
]
assert len(ICO_EDGES) == 30


def count_cross_edges_ico(phases):
    return sum(1 for i, j in ICO_EDGES if phases[i] != phases[j])


def ico_vertices(d):
    """12 icosahedron vertices at edge length d.

    MUST match the vertex ordering from studies/02b_icosahedron_phase2.py
    (golden ratio construction, unsorted, with even permutations of (0,+-1,+-phi)).
    """
    import numpy as np
    phi_gr = (1.0 + np.sqrt(5.0)) / 2.0
    base_verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            base_verts.append([0.0, s1 * 1.0, s2 * phi_gr])
            base_verts.append([s1 * 1.0, s2 * phi_gr, 0.0])
            base_verts.append([s1 * phi_gr, 0.0, s2 * 1.0])
    base_verts = np.array(base_verts)
    # Scale so minimum edge = d
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            dists.append(np.linalg.norm(base_verts[i] - base_verts[j]))
    min_edge = min(dists)
    scale = d / min_edge
    return (base_verts * scale).tolist()


# ============================================================
# Phase 7b configs: cube at d=7.5
# ============================================================
CUBE_D75_CONFIGS = [
    # (name, phases_as_signs, expected_CE)
    # phases: -1 = pi, +1 = 0
    ("all_same",       [+1, +1, +1, +1, +1, +1, +1, +1],  0),
    ("single_flip",    [-1, +1, +1, +1, +1, +1, +1, +1],  3),
    ("adjacent_2",     [-1, -1, +1, +1, +1, +1, +1, +1],  4),
    ("checkerboard",   [-1, -1, -1, +1, +1, +1, -1, +1],  6),
    ("polarized_T1",   [-1, +1, +1, -1, +1, -1, -1, +1], 12),
]

# d=6.0 reference E_bind values (from existing data)
D60_EBIND = {
    "all_same":      +73.8417,
    "single_flip":   +36.8866,
    "adjacent_2":    +20.5866,
    "checkerboard":   -3.8634,
    "polarized_T1":  -51.5264,
}

# ============================================================
# Worker function for a single config
# ============================================================

def run_single(args):
    """Run one simulation. args is a dict with all config info."""
    import numpy as np
    from scipy.interpolate import interp1d
    from engine.evolver import SexticEvolver
    from engine.checkpoint import run_with_checkpointing

    config = args
    name = config['name']
    output_path = config['output_path']

    # Check cache
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                cached = json.load(f)
            if cached.get('completed', False):
                print("CACHED: %s" % name)
                sys.stdout.flush()
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    # Load baseline for E_bind computation
    with open(PHASE1_JSON) as f:
        ref = json.load(f)
    E_single_interp = interp1d(ref["t_series"], ref["E_series"],
                                kind="linear", fill_value="extrapolate")

    # Build evolver
    N = config['N_grid']
    L = config['L']
    ev = SexticEvolver(N=N, L=L, m=config['m'], g4=config['g4'],
                       g6=config['g6'], dissipation_sigma=config['sigma_KO'])

    # Build initial conditions
    vertices = config['vertices']
    phases = config['phases']
    n_osc = len(vertices)

    phi_init = np.zeros((N, N, N))
    for idx in range(n_osc):
        pos = vertices[idx]
        A = phases[idx]  # +1 or -1
        dx_ = ev.X - pos[0]
        dy_ = ev.Y - pos[1]
        dz_ = ev.Z - pos[2]
        r2 = dx_**2 + dy_**2 + dz_**2
        phi_init += A * PHI0 * np.exp(-r2 / (2.0 * R_GAUSS**2))

    phi_dot_init = np.zeros_like(phi_init)
    ev.set_initial_conditions(phi_init, phi_dot_init)

    T_final = config['T_final']
    n_steps = int(T_final / DT)

    # Use run_with_checkpointing
    ckpt_config = {
        'name': name,
        'params': {
            'N_grid': N, 'L': L, 'm': config['m'],
            'g4': config['g4'], 'g6': config['g6'],
            'dt': DT, 'T_final': T_final, 'sigma_KO': config['sigma_KO'],
            'd_edge': config['d_edge'],
        },
        'metadata': {
            'geometry': config['geometry'],
            'config_name': name,
            'task': config.get('task', ''),
        },
        'initial_conditions': {
            'n_oscillons': n_osc,
            'phases': phases,
            'vertex_positions': vertices,
            'n_pi': sum(1 for p in phases if p < 0),
            'cross_edges': config['cross_edges'],
            'f_cross': config['f_cross'],
        },
        'record_every': RECORD_EVERY,
        'print_every': 2000,
    }

    result = run_with_checkpointing(ev, ckpt_config, output_path,
                                     checkpoint_interval=50.0)

    # Compute E_bind series post-hoc
    ts = result['time_series']
    times = ts['times']
    E_total = ts['E_total']

    # E_bind(t) = E_total(t) - n_oscillons * E_single(t)
    # For extended runs beyond T=500, extrapolate baseline
    E_bind_series = []
    for i in range(len(times)):
        t = times[i]
        E_bind_series.append(E_total[i] - n_osc * float(E_single_interp(min(t, 500.0))))

    # Add E_bind to result
    result['E_bind_series'] = E_bind_series
    result['final_E_bind'] = E_bind_series[-1]
    result['cross_edges'] = config['cross_edges']
    result['f_cross'] = config['f_cross']
    result['n_pi'] = sum(1 for p in phases if p < 0)
    result['verdict'] = "STABLE" if E_bind_series[-1] < -1.0 else "UNSTABLE"

    # For extended runs: compute diagnostics at milestones
    if T_final > 500:
        milestones = [500, 1000, 1500, 2000]
        milestone_data = {}
        for mt in milestones:
            if mt > T_final:
                break
            # Find closest time index
            best_idx = min(range(len(times)), key=lambda i: abs(times[i] - mt))
            milestone_data[str(mt)] = {
                'E_total': E_total[best_idx],
                'E_bind': E_bind_series[best_idx],
                'max_amplitude': ts['max_amplitude'][best_idx],
                'energy_drift_pct': abs(E_total[best_idx] - E_total[0]) / (abs(E_total[0]) + 1e-30),
                'amplitude_retention': ts['max_amplitude'][best_idx] / ts['max_amplitude'][0],
            }
        result['milestones'] = milestone_data

    # Atomic write with E_bind data added
    tmp_path = output_path + '.tmp'
    # Remove field arrays
    clean = dict(result)
    for key in ('phi_b64', 'phi_dot_b64', 'field_state'):
        clean.pop(key, None)
    clean['completed'] = True
    with open(tmp_path, 'w') as f:
        json.dump(clean, f, indent=2)
    os.replace(tmp_path, output_path)

    return result


# ============================================================
# Main
# ============================================================

def main():
    import numpy as np

    os.makedirs(OUTPUT_DIR_7B, exist_ok=True)
    os.makedirs(OUTPUT_DIR_7C, exist_ok=True)

    # ----------------------------------------------------------
    # Phase 7b: Verify cross-edge counts before building configs
    # ----------------------------------------------------------
    print("=" * 60)
    print("PHASE 7b: CUBE AT d=7.5 -- Cross-Edge Verification")
    print("=" * 60)

    d75_verts = cube_vertices(7.5)
    all_configs = []

    for name, phases, expected_ce in CUBE_D75_CONFIGS:
        actual_ce = count_cross_edges_cube(phases)
        status = "OK" if actual_ce == expected_ce else "MISMATCH"
        print("  %s: CE=%d (expected %d) -- %s" % (name, actual_ce, expected_ce, status))
        if actual_ce != expected_ce:
            print("  ERROR: Cross-edge mismatch for %s! Aborting." % name)
            sys.exit(1)

        all_configs.append({
            'name': 'cube_d75_%s' % name,
            'geometry': 'cube',
            'task': 'phase_7b',
            'output_path': os.path.join(OUTPUT_DIR_7B, 'cube_d75_%s.json' % name),
            'vertices': d75_verts,
            'phases': phases,
            'cross_edges': actual_ce,
            'f_cross': actual_ce / 12.0,
            'd_edge': 7.5,
            'T_final': 500.0,
            'N_grid': 64, 'L': 50.0, 'm': 1.0, 'g4': 0.30, 'g6': 0.055,
            'sigma_KO': 0.01,
        })

    print()
    sys.stdout.flush()

    # ----------------------------------------------------------
    # Phase 7c: Extended evolution configs
    # ----------------------------------------------------------
    print("=" * 60)
    print("PHASE 7c: EXTENDED EVOLUTION TO T=2000")
    print("=" * 60)

    # Cube polarized T1 at d=6.0
    d60_verts_cube = cube_vertices(6.0)
    cube_pol_phases = [-1, +1, +1, -1, +1, -1, -1, +1]  # CE=12
    ce_cube = count_cross_edges_cube(cube_pol_phases)
    print("  Cube polarized T1: CE=%d (expected 12)" % ce_cube)
    assert ce_cube == 12

    all_configs.append({
        'name': 'cube_polarized_T2000',
        'geometry': 'cube',
        'task': 'phase_7c',
        'output_path': os.path.join(OUTPUT_DIR_7C, 'cube_polarized_T2000.json'),
        'vertices': d60_verts_cube,
        'phases': cube_pol_phases,
        'cross_edges': 12,
        'f_cross': 1.0,
        'd_edge': 6.0,
        'T_final': 2000.0,
        'N_grid': 64, 'L': 50.0, 'm': 1.0, 'g4': 0.30, 'g6': 0.055,
        'sigma_KO': 0.01,
    })

    # Icosahedron ce_20 at d=6.0
    # Phases from outputs/phase2/icosahedron/ce_20.json: [-1,-1,1,-1,-1,1,1,1,-1,-1,1,1]
    ico_phases = [-1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1]
    ce_ico = count_cross_edges_ico(ico_phases)
    print("  Ico ce_20: CE=%d (expected 20)" % ce_ico)
    assert ce_ico == 20

    ico_verts = ico_vertices(6.0)

    all_configs.append({
        'name': 'ico_ce20_T2000',
        'geometry': 'icosahedron',
        'task': 'phase_7c',
        'output_path': os.path.join(OUTPUT_DIR_7C, 'ico_ce20_T2000.json'),
        'vertices': ico_verts,
        'phases': ico_phases,
        'cross_edges': 20,
        'f_cross': 20.0 / 30.0,
        'd_edge': 6.0,
        'T_final': 2000.0,
        'N_grid': 64, 'L': 50.0, 'm': 1.0, 'g4': 0.30, 'g6': 0.055,
        'sigma_KO': 0.01,
    })

    print()
    print("Total configs: %d (5 short T=500, 2 long T=2000)" % len(all_configs))
    print("Running with Pool(%d)..." % N_WORKERS)
    print("=" * 60)
    sys.stdout.flush()

    # ----------------------------------------------------------
    # Run all configs
    # ----------------------------------------------------------
    t_start = time.time()

    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, all_configs)

    wall_total = time.time() - t_start

    # ----------------------------------------------------------
    # Phase 7b summary
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("PHASE 7b RESULTS: CUBE AT d=7.5")
    print("=" * 60)

    d75_results = [r for r in results if r.get('metadata', {}).get('task') == 'phase_7b'
                   or 'cube_d75' in str(r.get('metadata', {}).get('config_name', ''))]

    # Build summary
    summary_7b = {
        'date': '2026-03-19',
        'description': 'Cube configs at d=7.5 (d/R_b=2.5), T=500, Set B parameters',
        'parameters': {
            'phi0': PHI0, 'R': R_GAUSS, 'm': 1.0, 'g4': 0.30, 'g6': 0.055,
            'N_grid': 64, 'L': 50.0, 'dt': DT, 'T_final': 500, 'sigma_KO': 0.01,
            'd_edge': 7.5,
        },
        'configs': [],
        'comparison_d60': [],
    }

    print()
    print("  %-18s %4s %8s %12s %12s %10s" % (
        "Config", "CE", "f_cross", "E_bind(d75)", "E_bind(d60)", "Verdict"))
    print("  " + "-" * 70)

    for cfg_def in CUBE_D75_CONFIGS:
        cfg_name = cfg_def[0]
        full_name = 'cube_d75_%s' % cfg_name
        # Find matching result
        r = None
        for res in results:
            rname = res.get('metadata', {}).get('config_name', '')
            if rname == full_name or full_name in str(res.get('metadata', {})):
                r = res
                break
        if r is None:
            print("  WARNING: no result for %s" % full_name)
            continue

        ebind = r.get('final_E_bind', r.get('E_bind_series', [None])[-1])
        ce = cfg_def[2]
        f_cross = ce / 12.0
        d60_eb = D60_EBIND.get(cfg_name, None)
        verdict = r.get('verdict', '?')

        print("  %-18s %4d %8.3f %12.4f %12.4f %10s" % (
            cfg_name, ce, f_cross,
            ebind if ebind is not None else 0,
            d60_eb if d60_eb is not None else 0,
            verdict))

        summary_7b['configs'].append({
            'name': cfg_name,
            'cross_edges': ce,
            'f_cross': f_cross,
            'E_bind_d75': ebind,
            'verdict': verdict,
        })
        summary_7b['comparison_d60'].append({
            'name': cfg_name,
            'E_bind_d60': d60_eb,
            'E_bind_d75': ebind,
            'sign_preserved': (d60_eb > 0) == (ebind > 0) if d60_eb is not None and ebind is not None else None,
        })

    # Check selection rule
    signs_match = all(c.get('sign_preserved', False) for c in summary_7b['comparison_d60'])
    summary_7b['selection_rule_holds'] = signs_match
    print()
    print("  Selection rule at d=7.5: %s" % ("HOLDS" if signs_match else "VIOLATED"))

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR_7B, 'summary.json')
    tmp = summary_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(summary_7b, f, indent=2)
    os.replace(tmp, summary_path)
    print("  Saved: %s" % summary_path)

    # ----------------------------------------------------------
    # Phase 7c summary
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("PHASE 7c RESULTS: EXTENDED EVOLUTION T=2000")
    print("=" * 60)

    for cfg_name in ['cube_polarized_T2000', 'ico_ce20_T2000']:
        r = None
        for res in results:
            rname = res.get('metadata', {}).get('config_name', '')
            if rname == cfg_name or cfg_name in str(res.get('metadata', {})):
                r = res
                break
        if r is None:
            print("  WARNING: no result for %s" % cfg_name)
            continue

        print()
        print("  %s:" % cfg_name)
        milestones = r.get('milestones', {})
        if milestones:
            print("  %6s %14s %14s %12s %12s" % (
                "T", "E_total", "E_bind", "Amp_Ret", "E_drift%"))
            print("  " + "-" * 60)
            for t_str in ['500', '1000', '1500', '2000']:
                if t_str in milestones:
                    m = milestones[t_str]
                    print("  %6s %14.4f %14.4f %12.4f %12.6f" % (
                        t_str, m['E_total'], m['E_bind'],
                        m['amplitude_retention'], m['energy_drift_pct']))
        else:
            ebind = r.get('final_E_bind', '?')
            print("  final E_bind = %s" % ebind)

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------
    from engine.checkpoint import cleanup_study
    cleanup_study(OUTPUT_DIR_7B)
    cleanup_study(OUTPUT_DIR_7C)

    print()
    print("=" * 60)
    print("ALL COMPLETE  (total wall time: %.1f min)" % (wall_total / 60.0))
    print("=" * 60)


if __name__ == '__main__':
    main()
