#!/bin/bash
# Reproduce all results from:
# "Geometric Selection Rules for Multi-Oscillon Cluster Stability"
# Estimated total: ~24 hours on Mac Mini (4 P-cores)
set -e
echo "=== Reproducing paper results ==="
echo "Started: $(date)"

echo "--- Step 1/11: Isolated baseline (Sec 2.3) ---"
python studies/01_single_reference.py

echo "--- Step 2/11: Pairwise cosine sweep (Table 1) ---"
python studies/10_continuous_phase.py

echo "--- Step 3/11: Cube binding (Table 2) ---"
python studies/02a_cube_phase2.py
python studies/02c_cube_adjacent_flip.py
python studies/cube_polarized_T1_CE12.py

echo "--- Step 4/11: Icosahedron binding (Table 2) ---"
python studies/02b_icosahedron_phase2.py

echo "--- Step 5/11: Tetrahedron + Octahedron (Table 2) ---"
python studies/04_new_geometries.py

echo "--- Step 6/11: Cross-edge universality (Table 2) ---"
python studies/03_cross_edge_universality.py

echo "--- Step 7/11: Ico CE=15 variants (Table 4) ---"
python studies/07b_ico_ce15_variants.py

echo "--- Step 8/11: Second potential (Table 3) ---"
python studies/09_second_potential.py

echo "--- Step 9/11: N=128 resolution (Sec 8.1) ---"
python studies/08_convergence_N128.py
python studies/09_selection_rule_N128.py

echo "--- Step 10/11: Spacing + long evolution (Sec 8.2-8.3) ---"
python studies/phase7bc_edge_length_and_extended.py

echo "--- Step 11/11: Self-organization ensemble (Table 5) ---"
echo "(This step takes ~14 hours)"
python studies/phase12_self_organization.py

echo "=== Complete ==="
echo "Finished: $(date)"
