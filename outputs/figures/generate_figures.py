"""
Publication-quality figure generation for swirl-theory project.
Generates Figure 1 (E_bind vs f_cross) and Figure 2 (convergence comparison).
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = '/Users/aaronramos/Desktop/swirl-theory'
OUT  = os.path.join(ROOT, 'outputs/figures')
os.makedirs(OUT, exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def load(path):
    with open(path) as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Collect central-figure data
# ─────────────────────────────────────────────────────────────────────────────

data_points = []   # list of dicts: geometry, label, f_cross, E_bind, jitter

# --- Tetrahedron (cross_edge_fraction already stored) ---
tet_dir = os.path.join(ROOT, 'outputs/phase4_new_geometries/tetrahedron')
tet_files = {
    'all_same':          'all_same.json',
    'single_flip':       'single_flip.json',
    'two_flip_adjacent': 'two_flip_adjacent.json',
    'two_flip_opposite': 'two_flip_opposite.json',
}
for label, fname in tet_files.items():
    d = load(os.path.join(tet_dir, fname))
    data_points.append(dict(
        geometry='Tetrahedron',
        label=label,
        f_cross=float(d['cross_edge_fraction']),
        E_bind=float(d['final_E_bind']),
        jitter=0.0,
    ))

# --- Octahedron (cross_edge_fraction already stored) ---
oct_dir = os.path.join(ROOT, 'outputs/phase4_new_geometries/octahedron')
oct_files = {
    'all_same':    'all_same.json',
    'single_flip': 'single_flip.json',
    'balanced':    'balanced.json',
    'polarized':   'polarized.json',
}
for label, fname in oct_files.items():
    d = load(os.path.join(oct_dir, fname))
    data_points.append(dict(
        geometry='Octahedron',
        label=label,
        f_cross=float(d['cross_edge_fraction']),
        E_bind=float(d['final_E_bind']),
        jitter=0.0,
    ))

# --- Cube (12 total edges; compute f_cross from cross_edges) ---
cube_dir = os.path.join(ROOT, 'outputs/phase2/cube')
cube_files = {
    'all_same':   'all_same_phase.json',
    'single_flip':'single_flip.json',
    'adj_flip':   'adjacent_flip.json',
    'checker':    'checkerboard.json',
    '4flip_CE8':  'polarized_T1.json',
    'polar_T1':   'cube_polarized_T1_CE12.json',
}
CUBE_EDGES = 12
for label, fname in cube_files.items():
    d = load(os.path.join(cube_dir, fname))
    f = d['cross_edges'] / CUBE_EDGES
    data_points.append(dict(
        geometry='Cube',
        label=label,
        f_cross=f,
        E_bind=float(d['final_E_bind']),
        jitter=0.0,
    ))

# --- Icosahedron (30 total edges) ---
ico_dir = os.path.join(ROOT, 'outputs/phase2/icosahedron')
ICO_EDGES = 30
for fname in sorted(os.listdir(ico_dir)):
    if not fname.endswith('.json') or fname == 'summary.json':
        continue
    d = load(os.path.join(ico_dir, fname))
    f = d['cross_edges'] / ICO_EDGES
    # Skip ce_15 from main series; will add variants below with jitter
    if fname == 'ce_15.json':
        continue
    data_points.append(dict(
        geometry='Icosahedron',
        label=fname.replace('.json', ''),
        f_cross=f,
        E_bind=float(d['final_E_bind']),
        jitter=0.0,
    ))

# --- Icosahedron ce_15 variants (jittered) ---
ico15_dir = os.path.join(ROOT, 'geometric_binding_study/outputs/icosahedron_threshold')
jitter_offsets = {'ico_ce_15_A.json': -0.005, 'ico_ce_15_B.json': 0.0, 'ico_ce_15_C.json': 0.005}
for fname, jitter in jitter_offsets.items():
    d = load(os.path.join(ico15_dir, fname))
    f_cross = float(d['initial_conditions']['f_cross'])
    E_bind  = float(d['final_state']['E_bind_final'])
    label   = d['metadata']['config_name']
    data_points.append(dict(
        geometry='Icosahedron',
        label=label,
        f_cross=f_cross,
        E_bind=E_bind,
        jitter=jitter,
    ))

# Print loaded data for verification
print("=== Central Figure Data ===")
for p in sorted(data_points, key=lambda x: (x['geometry'], x['f_cross'])):
    print(f"  {p['geometry']:12s}  {p['label']:22s}  f={p['f_cross']:.4f}  E_bind={p['E_bind']:+.4f}  jitter={p['jitter']:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Save central-figure data JSON
# ─────────────────────────────────────────────────────────────────────────────
out_json = os.path.join(OUT, 'central_figure_data.json')
with open(out_json, 'w') as fh:
    json.dump(data_points, fh, indent=2)
print(f"\nSaved data JSON -> {out_json}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Figure 1: E_bind vs f_cross
# ─────────────────────────────────────────────────────────────────────────────

GEOM_STYLE = {
    'Tetrahedron': dict(marker='^', color='#1f77b4', label='Tetrahedron'),
    'Octahedron':  dict(marker='s', color='#ff7f0e', label='Octahedron'),
    'Cube':        dict(marker='D', color='#2ca02c', label='Cube'),
    'Icosahedron': dict(marker='o', color='#d62728', label='Icosahedron'),
}

fig, ax = plt.subplots(figsize=(8, 5))

# Grid
ax.grid(True, color='lightgray', linewidth=0.6, zorder=0)
ax.set_axisbelow(True)

# Threshold band
ax.axvspan(0.47, 0.52, color='lightgray', alpha=0.30, zorder=1, label='_nolegend_')

# Zero line
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=2)

# Plot each geometry as a separate scatter series
plotted_geoms = set()
for geom, style in GEOM_STYLE.items():
    pts = [p for p in data_points if p['geometry'] == geom]
    if not pts:
        continue
    xs = [p['f_cross'] + p['jitter'] for p in pts]
    ys = [p['E_bind'] for p in pts]
    kw = dict(
        marker=style['marker'],
        color=style['color'],
        s=60,
        linewidths=0.8,
        edgecolors='black',
        zorder=4,
    )
    if geom not in plotted_geoms:
        ax.scatter(xs, ys, label=style['label'], **kw)
        plotted_geoms.add(geom)
    else:
        ax.scatter(xs, ys, **kw)

# Axis labels and formatting
ax.set_xlabel(r'Cross-edge fraction $f_{\mathrm{cross}}$', fontsize=12)
ax.set_ylabel(r'Binding energy $E_{\mathrm{bind}}$', fontsize=12)
ax.tick_params(labelsize=10)
ax.set_xlim(-0.02, 1.02)

# Legend
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

fig.tight_layout()

png_path = os.path.join(OUT, 'central_figure.png')
pdf_path = os.path.join(OUT, 'central_figure.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
plt.close(fig)
print(f"Saved Figure 1 -> {png_path}")
print(f"Saved Figure 1 -> {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Convergence data
# ─────────────────────────────────────────────────────────────────────────────
# N128 values come from individual convergence JSON files.
# N64 values come from selection_rule_convergence.json and phase data files.

# Labels for x-axis (same order)
CONV_CONFIGS = [
    ('tet_1flip',      'Tet 1-flip',       -0.041,          -0.04082803508971722    ),
    ('cube_allsame',   'Cube all-same',    73.8417,          73.84819301919327       ),
    ('cube_checker',   'Cube checker',     -3.8634,          -3.863649510003711      ),
    ('cube_polar_T1',  'Cube pol-T1',      -51.526,          -51.52931854790001      ),
    ('ico_ce15A',      'Ico ce15A',         3.557,             3.557598268460879      ),
    ('ico_ce20',       'Ico ce20',         -50.1083,         -50.111624451185264     ),
]

labels_conv  = [c[1] for c in CONV_CONFIGS]
E_bind_N64   = [c[2] for c in CONV_CONFIGS]
E_bind_N128  = [c[3] for c in CONV_CONFIGS]

print("\n=== Convergence Data ===")
for cfg, lbl, e64, e128 in CONV_CONFIGS:
    print(f"  {lbl:18s}  N64={e64:+.4f}  N128={e128:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Figure 2: Convergence bar chart
# ─────────────────────────────────────────────────────────────────────────────

x = np.arange(len(labels_conv))
width = 0.35

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.grid(True, axis='y', color='lightgray', linewidth=0.6, zorder=0)
ax2.set_axisbelow(True)

bars64  = ax2.bar(x - width/2, E_bind_N64,  width, label='N=64',  color='#1f77b4',
                  edgecolor='black', linewidth=0.6, zorder=3)
bars128 = ax2.bar(x + width/2, E_bind_N128, width, label='N=128', color='#ff7f0e',
                  edgecolor='black', linewidth=0.6, zorder=3)

ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=4)

ax2.set_xticks(x)
ax2.set_xticklabels(labels_conv, fontsize=10, rotation=15, ha='right')
ax2.set_ylabel(r'Binding energy $E_{\mathrm{bind}}$', fontsize=12)
ax2.tick_params(axis='y', labelsize=10)
ax2.legend(fontsize=10, framealpha=0.9)

fig2.tight_layout()

png2_path = os.path.join(OUT, 'convergence_comparison.png')
pdf2_path = os.path.join(OUT, 'convergence_comparison.pdf')
fig2.savefig(png2_path, dpi=300, bbox_inches='tight')
fig2.savefig(pdf2_path, bbox_inches='tight')
plt.close(fig2)
print(f"\nSaved Figure 2 -> {png2_path}")
print(f"Saved Figure 2 -> {pdf2_path}")
print("\nDone.")
