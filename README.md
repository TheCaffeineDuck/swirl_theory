# Geometric Selection Rules for Multi-Oscillon Cluster Stability

Code and data for the paper by Aaron Choi Ramos (April 2026).

## Abstract

We present a systematic numerical study of multi-oscillon cluster stability in 3+1D scalar field theories with sextic self-interaction. For clusters of 4 to 12 oscillons arranged at the vertices of Platonic solids, the binding energy is governed by the phase structure of nearest-neighbor pairs. We establish three principal results, building on the analytically predicted cosine interaction law.

First, we verify that the pairwise interaction energy follows E_pair(delta_phi) = A cos(delta_phi) + B with R^2 = 0.999999, yielding a generalized selection rule: a cluster is bound when sum of cos(delta_phi_e) < 0. Second, the resulting 50% cross-edge fraction threshold is universal across potentials. Third, at the threshold, neighbor signature variance is a deterministic secondary predictor (R^2 = 0.997).

A 50-seed ensemble with random continuous initial phases demonstrates that the selection rule operates as a dynamical attractor (z = -15.4), with 60% achieving negative binding energy within 177 oscillation periods.

## Dependencies

- Python 3.10+
- NumPy, SciPy, Numba, Matplotlib

```bash
pip install -r requirements.txt
```

## Reproducing Results

Run all paper results in sequence (~24 hours on 4 P-cores):
```bash
bash studies/reproduce_all.sh
```

Individual tables:

| Paper Section | Script | Est. Time |
|---|---|---|
| Sec 2.3 — Isolated baseline | `studies/01_single_reference.py` | 5 min |
| Table 1 — Pairwise cosine | `studies/10_continuous_phase.py` | 30 min |
| Table 2 — Cube binding | `studies/02a_cube_phase2.py`, `02c_*`, `cube_polarized_T1_CE12.py` | 45 min |
| Table 2 — Icosahedron | `studies/02b_icosahedron_phase2.py` | 60 min |
| Table 2 — Tet + Oct | `studies/04_new_geometries.py` | 30 min |
| Table 2 — Cross-edge analysis | `studies/03_cross_edge_universality.py` | 20 min |
| Table 3 — Set C universality | `studies/09_second_potential.py` | 45 min |
| Table 4 — Variance rule | `studies/07b_ico_ce15_variants.py` | 45 min |
| Sec 8.1 — N=128 resolution | `studies/08_convergence_N128.py`, `09_selection_rule_N128.py` | 4 hrs |
| Sec 8.2–8.3 — Spacing + T=2000 | `studies/phase7bc_edge_length_and_extended.py` | 3 hrs |
| Table 5 — Self-organization | `studies/phase12_self_organization.py` | 14 hrs |

## Parameter Sets

| Parameter | Set B (primary) | Set C (universality) |
|---|---|---|
| m | 1.0 | 1.0 |
| g4 | 0.30 | 0.50 |
| g6 | 0.055 | 0.10 |
| phi0 | 0.5 | 0.5 |
| R | 2.5 | 2.5 |

Grid: N=64, L=50.0, dt=0.05, d=6.0, sigma_KO=0.01

## Repository Structure

```
engine/          — SexticEvolver simulation engine (spectral RK4, sextic potential)
studies/         — Scripts reproducing each paper table/section
configs/         — Default parameter files
outputs/         — Simulation results (JSON)
paper/           — Paper draft
docs/            — Compute guidelines, literature review
```

## Hardware

Developed on Mac Mini (Apple Silicon, 4 performance cores).
See `docs/COMPUTE_GUIDELINES.md` for parallelization details.

## License

MIT
