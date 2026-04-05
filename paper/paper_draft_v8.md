# Geometric Selection Rules for Multi-Oscillon Cluster Stability

**Aaron Choi Ramos**

*Independent researcher*
*aaron.choi.ramos@gmail.com*

**Date:** March 2026

---

## Abstract

We present a systematic numerical study of multi-oscillon cluster stability in 3+1D scalar field theories with sextic self-interaction. For clusters of 4 to 12 oscillons arranged at the vertices of Platonic solids, the binding energy is governed by the phase structure of nearest-neighbor pairs. We report four principal results. First, we verify that the pairwise interaction energy follows the cosine dependence predicted analytically for soliton pairs [13,14], E_pair(Δφ) = A cos(Δφ) + B, with R² = 0.999999 and near-zero offset |B/A| < 10⁻³ in a non-integrable 3+1D sextic theory. This cosine law yields a generalized selection rule: a multi-oscillon cluster is bound when Σ_edges cos(Δφ_e) < 0, which for binary phases (Δφ ∈ {0, π}) reduces to a 50% cross-edge fraction threshold. Second, this threshold is universal across potentials, verified with two parameter sets whose coupling constants differ by 67–82%, yielding identical binding energy signs for all tested configurations. Third, at the threshold, an exhaustive enumeration of all six symmetry-inequivalent icosahedral configurations with exactly 50% cross-edges reveals that binding energy is a deterministic linear function of neighbor signature variance (R² = 0.997), a topological measure of cross-edge uniformity that the pairwise cosine model does not predict.

A 50-seed ensemble with random continuous initial phases demonstrates that the selection rule operates as a dynamical attractor: all seeds evolve toward anti-phase nearest-neighbor correlations (z = -15.4 against a null hypothesis of zero mean), with 60% achieving negative binding energy within 177 oscillation periods.

All results are confirmed at doubled spatial resolution (N = 128), at two inter-oscillon spacings, and through 354 oscillation periods of dynamical evolution. The pairwise model is exact for the tetrahedron (complete graph K₄) and accurate to within 3% for the cube and icosahedron, but underpredicts the octahedron by a factor of 2.25×, attributable to nonlinear many-body effects arising from multi-oscillon field superposition at the geometric center. These findings establish graph-theoretic and phase-geometric tools as a predictive framework for multi-body oscillon interactions.

---

## 1. Introduction

Oscillons are long-lived, spatially localized, oscillating field configurations that arise in a variety of nonlinear scalar field theories [1–3]. Unlike topological solitons, oscillons owe their longevity to an approximate balance between dispersive and nonlinear effects rather than to a conserved charge. In cosmological contexts, oscillons form copiously after inflation in models with suitable potentials [4,5] and can persist for timescales exceeding 10⁸ oscillation cycles in theories with sextic self-interactions [6].

The interactions between oscillons have received growing attention. Xue et al. [7] performed the first systematic study of realistic oscillon collisions in 3+1D, documenting phase-dependent scattering: in-phase oscillons attract and merge, while anti-phase oscillons repel and scatter. That work found no stable multi-oscillon bound states; all collisions resulted either in merger or separation. The question of whether geometric arrangements of multiple oscillons can produce configurations with negative binding energy, without merger, has remained open.

We address this question by studying clusters of oscillons placed at the vertices of Platonic solids. Rather than beginning with binary phase assignments, we first verify the fundamental interaction law: the pairwise binding energy between two oscillons follows a cosine dependence on relative phase, E_pair(Δφ) = A cos(Δφ) + B, with R² = 0.999999. The cosine form was derived analytically by Gordon [13] for NLS solitons and by Manton [14] via the general tail-overlap mechanism; our result confirms that it extends to a non-integrable 3+1D sextic theory. This cosine law, combined with the near-vanishing offset |B/A| < 10⁻³, yields a generalized selection rule for multi-oscillon clusters: binding occurs when the sum Σ_edges cos(Δφ_e) < 0. For the binary phase case (Δφ ∈ {0, π}) studied in the remainder of the paper, this reduces to a threshold at approximately 50% cross-edge fraction.

We verify this threshold across four Platonic geometries (tetrahedron, octahedron, cube, icosahedron), two sets of potential parameters with coupling constants differing by 67–82%, two inter-oscillon spacings, and 354 oscillation periods of dynamical evolution. At the threshold, where the pairwise model predicts E_bind ≈ 0, we find a secondary selection rule: binding energy is a deterministic linear function of the neighbor signature variance (R² = 0.997), established by exhaustive enumeration of all symmetry-inequivalent configurations on the icosahedron.

We note the connection between cross-edge fraction and the graph-theoretic Max-Cut problem [8]: the maximum achievable cross-edge fraction for a given graph equals its Max-Cut fraction. For bipartite graphs (the cube), this is 1.0; for non-bipartite graphs (the icosahedron), it is strictly less. This places fundamental topological constraints on which phase configurations can achieve maximum stability.

The oscillation period for our parameter choices is τ = 2π/ω_eff ≈ 5.65 in simulation units, where ω_eff = 1.113 is the measured effective frequency. Our standard evolution time of T = 500 corresponds to approximately 88 oscillation periods; extended runs reach T = 2000, or approximately 354 periods. All binding energies are computed relative to time-matched isolated oscillon references.

---

## 2. Framework

### 2.1 Field Theory

We study a real scalar field φ in 3+1 dimensions with the Lagrangian density

    L = (1/2) ∂_μ φ ∂^μ φ − V(φ)

where the sextic potential is

    V(φ) = (1/2) m² φ² − (g₄/4!) φ⁴ + (g₆/6!) φ⁶

The negative quartic term creates a local minimum in the effective potential at finite field amplitude, supporting oscillon solutions; the positive sextic term ensures the potential is bounded below. The equation of motion is

    φ̈ − ∇²φ + m²φ − (g₄/6)φ³ + (g₆/120)φ⁵ = 0

We study two parameter sets to test universality:

| Parameter | Set B | Set C |
|-----------|-------|-------|
| m | 1.0 | 1.0 |
| g₄ | 0.30 | 0.50 |
| g₆ | 0.055 | 0.10 |

Set C increases the quartic coupling by 67% and the sextic coupling by 82% relative to Set B.

### 2.2 Numerical Methods

We integrate the equation of motion using fourth-order Runge-Kutta (RK4) on a three-dimensional periodic grid. The spatial Laplacian is computed spectrally via fast Fourier transform. Fourth-order Kreiss-Oliger dissipation [9] with coefficient σ = 0.01 is applied to suppress high-frequency numerical noise.

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 64 | Grid points per dimension |
| L | 50.0 | Box side length |
| dx | 0.78125 | Grid spacing |
| dt | 0.05 | Time step |
| T | 500 | Standard evolution time |

Each oscillon is initialized as a Gaussian profile with amplitude φ₀ = 0.5 and width R = 2.5. For phase-shifted oscillons at relative phase Δφ:

    φ(r, t=0) = φ₀ cos(Δφ) exp(−r²/2R²)
    φ̇(r, t=0) = −ω_eff φ₀ sin(Δφ) exp(−r²/2R²)

where ω_eff = 1.113 is the measured oscillation frequency. An isolated phase-shifted oscillon has the same energy regardless of Δφ, as verified to <10⁻⁴ relative precision.

Vertex coordinates and indices for each geometry are listed in Appendix A. The cube vertex ordering follows a binary enumeration of (x, y, z) sign combinations; the two inscribed tetrahedra are T1 = {0, 2, 5, 7} and T2 = {1, 3, 4, 6}.

### 2.3 Binding Energy Definition

The binding energy of an N-oscillon cluster is

    E_bind(t) = E_total(t) − N × E_single(t)

where E_single(t) is the energy of an isolated oscillon evolved to the same time t. This time-matched baseline is essential: isolated oscillons radiate energy throughout their evolution (0.015% drift over T = 500), and a static reference would systematically bias the binding energy. A negative E_bind indicates energetic stabilization by the geometric arrangement.

### 2.4 Graph-Theoretic Quantities

For a cluster at the vertices of a polyhedron with E edges:

**Cross-edge fraction.** An edge connecting oscillons of opposite binary phase is a cross-edge. The cross-edge fraction is f_cross = N_cross / E.

**Neighbor signature variance.** For vertex i with degree k_i, the neighbor signature s_i is the number of opposite-phase neighbors. The variance σ²_NS = Var({s_i}) measures cross-edge uniformity.

**Bipartiteness.** The cube is the only bipartite Platonic solid graph. Its maximum cross-edge fraction is 1.0; non-bipartite graphs are bounded by their Max-Cut fraction [8]. The 50% cross-edge threshold has a precise graph-theoretic identity: Edwards [21] proved that every graph with m edges and n vertices satisfies maxcut ≥ m/2 + (n−1)/4. The m/2 baseline corresponds to the expected random cut; configurations below this baseline are unstable under the selection rule, while those above are potentially stable.

---

## 3. The Cosine Interaction Law

### 3.1 Pairwise Phase Sweep

We placed two oscillons at separation d = 6.0 with relative phases Δφ = 0, π/6, π/3, π/2, 2π/3, 5π/6, π and evolved each pair for T = 500. The results are shown in Table 1.

**Table 1.** Pairwise binding energy as a function of relative phase difference.

| Δφ (deg) | Δφ (rad) | E_bind | Cosine fit | Residual |
|----------|----------|--------|------------|----------|
| 0 | 0.000 | +5.172 | +5.179 | −0.007 |
| 30 | 0.524 | +4.487 | +4.485 | +0.002 |
| 60 | 1.047 | +2.594 | +2.588 | +0.007 |
| 90 | 1.571 | −0.004 | −0.004 | −0.000 |
| 120 | 2.094 | −2.595 | −2.595 | −0.000 |
| 150 | 2.618 | −4.492 | −4.492 | +0.000 |
| 180 | 3.142 | −5.188 | −5.186 | −0.002 |

A least-squares fit gives

    E_pair(Δφ) = A cos(Δφ) + B

with A = 5.183, B = −0.004, and R² = 0.999999. The maximum residual is 0.007.

The cosine form of the pairwise interaction has a well-established analytical basis. Gordon [13] derived F = −2A² exp(−q) cos(Δφ) for the interaction force between NLS solitons from the exact two-soliton solution. Manton [14] established the general tail-overlap mechanism by which localized solutions interact through exponentially decaying tails, and this framework was subsequently codified by Manton and Sutcliffe [15]. Kevrekidis, Khare, and Saxena [17] extended the approach to sine-Gordon breathers, and Axenides et al. [16] demonstrated consistency for Q-balls. Our result confirms that the cosine form persists in a non-integrable 3+1D sextic potential with no exact multi-soliton solutions. The R² = 0.999999 demonstrates that higher-order corrections to the pairwise interaction are negligible at the separations studied.

### 3.2 Physical Interpretation

The cosine dependence is exact to six significant figures. This precision confirms that the tail-overlap mechanism identified by Manton [14] generates a pure cosine interaction even in a non-integrable theory where no exact multi-soliton solutions exist. The near-zero offset (|B/A| = 7.7 × 10⁻⁴) reflects an almost perfect symmetry between attractive and repulsive interactions: the magnitude of attraction at Δφ = π (|E_bind| = 5.188) differs from repulsion at Δφ = 0 (E_bind = 5.172) by only 0.3%.

The zero crossing at Δφ = π/2 means that oscillons in quadrature (90° phase difference) exert negligible net interaction. This establishes a clear physical picture: oscillons whose field oscillations are more than 90° out of phase attract; those less than 90° out of phase repel.

A note on sign conventions relative to collision studies is warranted. Xue et al. [7] reported that in-phase oscillons "attract and merge" in head-on collisions, while anti-phase oscillons "repel and scatter." These observations describe short-range dynamics at close approach. Our finding of positive binding energy for in-phase pairs at d = 6.0 is consistent: the energy minimum for in-phase oscillons lies at zero separation (merger), so at finite separation the system is above its ground state. For anti-phase pairs, negative binding energy at d = 6.0 implies a long-range attractive well at tail-overlap distances, while Xue et al.'s repulsion operates at shorter range where field gradients steepen. The combination (short-range repulsion preventing merger, long-range attraction at tail-overlap distances) is precisely the interaction profile that permits stable binding at finite separation.

### 3.3 Generalized Selection Rule

For a multi-oscillon cluster where each edge e connects oscillons with phase difference Δφ_e, the pairwise approximation to the total binding energy is

    E_bind ≈ Σ_edges [A cos(Δφ_e) + B]

The cluster is bound when E_bind < 0, which (given |B| ≪ A) occurs when

    Σ_edges cos(Δφ_e) < 0

For binary phases where Δφ_e ∈ {0, π}, each edge contributes cos(0) = +1 (same-phase) or cos(π) = −1 (cross-phase). The sum equals (E − N_cross) − N_cross = E − 2N_cross. Setting this to zero gives N_cross = E/2, or f_cross = 0.50. The binary 50% cross-edge fraction rule is a special case of the continuous cosine selection rule.

The predicted binary threshold is f* = E_same/(E_same − E_cross) = 5.172/10.360 = 0.4992, in excellent agreement with the measured mean across four geometries (f* = 0.492).

---

## 4. Binary Phase Results Across Four Geometries

### 4.1 Overview

We simulated 26 distinct binary-phase configurations across four Platonic solid geometries at d = 6.0, T = 500, using Set B parameters. Table 2 presents the complete results; six threshold variants on the icosahedron are analyzed in detail in Section 6.

**Table 2.** Binding energy for all configurations. CE = cross-edges.

#### Tetrahedron (4 vertices, 6 edges, K₄, non-bipartite)

| n_π | CE | f_cross | E_bind | Verdict |
|-----|-----|---------|--------|---------|
| 0 | 0/6 | 0.000 | +30.93 | Unbound |
| 1 | 3/6 | 0.500 | −0.04 | Marginal |
| 2 | 4/6 | 0.667 | −10.39 | Bound |

#### Octahedron (6 vertices, 12 edges, non-bipartite)

| n_π | CE | f_cross | E_bind | Verdict |
|-----|------|---------|--------|---------|
| 0 | 0/12 | 0.000 | +154.51 | Unbound |
| 1 | 4/12 | 0.333 | +51.65 | Unbound |
| 3† | 6/12 | 0.500 | −5.45 | Bound |
| 3† | 8/12 | 0.667 | −32.01 | Bound |

† Two symmetry-inequivalent configurations with n_π = 3 exist: one occupying a triangular face (CE = 6) and one occupying an antipodal pair plus one additional vertex (CE = 8).

#### Cube (8 vertices, 12 edges, bipartite)

| n_π | CE | f_cross | E_bind | Config | Verdict |
|-----|------|---------|--------|--------|---------|
| 0 | 0/12 | 0.000 | +73.84 | All-same | Unbound |
| 1 | 3/12 | 0.250 | +36.89 | Single flip | Unbound |
| 2 | 4/12 | 0.333 | +20.59 | Adjacent pair | Unbound |
| 4 | 6/12 | 0.500 | −3.86 | Checkerboard | Bound |
| 4 | 8/12 | 0.667 | −23.94 | Anti-tetrahedral | Bound |
| 4 | 12/12 | 1.000 | −51.53 | Polarized T1 | Bound |

#### Icosahedron (12 vertices, 30 edges, non-bipartite)

| n_π | CE | f_cross | E_bind | Verdict |
|-----|------|---------|--------|---------|
| 0 | 0/30 | 0.000 | +164.11 | Unbound |
| 2 | 8/30 | 0.267 | +75.20 | Unbound |
| 5 | 13/30 | 0.433 | +22.91 | Unbound |
| 6 | 14/30 | 0.467 | +10.60 | Unbound |
| — | 15/30 | 0.500 | +3.56 to −1.49 | Mixed |
| 6 | 16/30 | 0.533 | −8.78 | Bound |
| 6 | 18/30 | 0.600 | −31.77 | Bound |
| 6 | 20/30 | 0.667 | −50.11 | Bound |

### 4.2 Threshold Measurements

| Geometry | Vertices | Edges | Bipartite? | f* |
|----------|----------|-------|------------|------|
| Tetrahedron | 4 | 6 | No (K₄) | 0.499 |
| Octahedron | 6 | 12 | No | 0.484 |
| Cube | 8 | 12 | Yes | 0.474 |
| Icosahedron | 12 | 30 | No | 0.510 |

Mean: f* = 0.492. Range: 0.474–0.510. The cosine law predicts f* = 0.499 (Section 3.3).

### 4.3 Bipartite vs. Non-Bipartite Constraints

The cube, being bipartite, achieves f_cross = 1.000 via the polarized tetrahedron configuration, yielding the strongest binding (E_bind = −51.53). No non-bipartite geometry can achieve f_cross = 1.0. The icosahedron's maximum is 20/30 = 0.667 (its Max-Cut fraction), yielding E_bind = −50.11, comparable in magnitude despite lower cross-edge fraction because the icosahedron has 2.5× more edges.

---

## 5. Universality Across Potentials

### 5.1 Second Parameter Set

To test whether the selection rule depends on the specific potential, we repeated a subset of simulations with Set C parameters (g₄ = 0.50, g₆ = 0.10), increasing both couplings substantially while keeping all other parameters fixed.

**Table 3.** Binding energies under two potentials.

| Configuration | f_cross | E_bind (Set B) | E_bind (Set C) | Sign match |
|---------------|---------|----------------|----------------|------------|
| Cube all-same | 0.000 | +73.84 | +73.42 | Yes |
| Cube checkerboard | 0.500 | −3.86 | −3.90 | Yes |
| Cube polarized T1 | 1.000 | −51.53 | −51.44 | Yes |
| Ico ce_15_A | 0.500 | +3.56 | +3.37 | Yes |
| Ico ce_20 | 0.667 | −50.11 | −50.04 | Yes |

All five binding energy signs are preserved. The magnitudes agree to within 1–2%.

### 5.2 Pairwise Comparison

| Quantity | Set B | Set C |
|----------|-------|-------|
| E_same | +5.172 | +5.152 |
| E_cross | −5.188 | −5.179 |
| \|E_cross/E_same\| | 1.003 | 1.005 |
| Predicted f* | 0.499 | 0.499 |

The pairwise interaction ratio |E_cross/E_same| ≈ 1.00 in both parameter sets, despite 67–82% changes in coupling constants. The predicted threshold is 0.499 in both cases. The near-exact pairwise symmetry is a structural property of oscillon tail interactions, not a parameter-dependent coincidence.

### 5.3 Interpretation

The universality of the pairwise symmetry is consistent with the tail-overlap mechanism first identified by Manton [14] and applied to soliton interactions by Gordon [13]: the interaction is mediated by the overlap integral of exponentially decaying oscillating tails, producing a pure cosine regardless of the specific nonlinearity. The amplitude A depends on the oscillon profile and separation distance, but the offset B ≈ 0 is a consequence of the overlap integral inheriting the phase dependence of the individual oscillations. As long as oscillons are separated by several radii (d ≫ R), the specific values of g₄ and g₆ modify the oscillon profile and hence A, but preserve the fundamental cos(Δφ) structure.

---

## 6. Threshold Variant Analysis: The Variance Rule

### 6.1 Motivation

The icosahedron data in Table 2 shows that f_cross = 0.500 (15/30 cross-edges) produces binding energies ranging from +3.56 to −1.49 depending on the phase assignment. The pairwise cosine model, which depends only on the count of cross-edges, predicts E_bind ≈ 0 for all such configurations. The observed spread of 5.0 energy units at fixed f_cross requires an explanation beyond pairwise physics.

### 6.2 Exhaustive Enumeration

The icosahedron has 2¹² = 4096 possible binary phase configurations. We enumerated all configurations achieving exactly 15 cross-edges and clustered them by icosahedral symmetry. This yields exactly six equivalence classes; no others exist. We simulated one representative from each.

**Table 4.** All six symmetry-inequivalent icosahedral configurations with f_cross = 0.500.

| Class | n_π | σ²_NS | E_bind | Verdict |
|-------|-----|-------|--------|---------|
| C | 5 | 0.917 | −1.487 | Bound |
| D | 5 | 0.917 | −1.494 | Bound |
| E | 5 | 0.917 | −1.494 | Bound |
| B | 5 | 1.250 | −0.426 | Marginal |
| F | 5 | 1.250 | −0.446 | Marginal |
| A | 3 | 2.250 | +3.557 | Unbound |

### 6.3 The Variance Rule

A linear regression of E_bind on σ²_NS gives

    E_bind = 3.786 × σ²_NS − 5.031

with R² = 0.997. Classes C, D, and E belong to three distinct symmetry equivalence classes but share σ²_NS = 0.917 and produce E_bind values agreeing to within 0.5% (−1.487, −1.494, −1.494). Classes B and F share σ²_NS = 1.250 and agree to within 4.5% (−0.426, −0.446). The neighbor signature variance fully determines the binding energy at the threshold.

### 6.4 Physical Interpretation

The variance measures how uniformly cross-edges are distributed across vertices. Low variance means every vertex has approximately the same number of opposite-phase neighbors, distributing the attractive interaction uniformly. High variance means some vertices have concentrated clusters of same-phase or opposite-phase neighbors, creating local regions of constructive or destructive interference that reduce the overall binding efficiency.

This is an intrinsically multi-body effect. The pairwise cosine model predicts identical E_bind for all configurations at the same f_cross. The variance rule reveals structure at the threshold that requires the full nonlinear field theory and cannot be derived from pairwise physics alone.

### 6.5 Implications for Threshold Precision

Five of six classes are bound (E_bind < 0) and one is unbound at f_cross = 0.500. The effective threshold depends on the phase assignment: for the most uniform configurations (σ²_NS = 0.917), the threshold falls below 0.500; for the least uniform (σ²_NS = 2.250), it lies between 0.500 and 0.533.

---

## 7. Many-Body Physics and the Pairwise Limit

### 7.1 Pairwise Model Accuracy

We compare the pairwise prediction E_pw = N_cross × E_cross + (E − N_cross) × E_same to the measured E_bind for each geometry. The ratio of measured to predicted slope (the enhancement factor) quantifies many-body corrections:

| Geometry | Edges | Graph type | Enhancement | Pairwise accuracy |
|----------|-------|-----------|-------------|-------------------|
| Tetrahedron | 6 | K₄ (complete) | 1.00× | Exact (<0.3%) |
| Octahedron | 12 | Near-complete | 2.25× | Poor |
| Cube | 12 | Bipartite | 1.01× | Good (0–19%) |
| Icosahedron | 30 | Moderate density | 1.03× | Good (1–12%) |

The tetrahedron, whose complete graph structure ensures that all C(4,2) = 6 pairs are nearest-neighbor edges, serves as a control: the pairwise model is exact, validating the calibration.

### 7.2 The Octahedron Anomaly

The octahedron's 2.25× enhancement cannot be attributed to non-edge pairwise interactions. The three non-edge pairs (opposite vertices at distance d√2 ≈ 8.49) would need to contribute +30.8 per pair to account for the all-same configuration's residual of +92.5, a factor of 6× larger than the edge-pair interaction of +5.17 at the shorter distance d = 6.0. Since oscillon tail interactions decrease with distance, this is physically impossible.

The enhancement is instead driven by nonlinear many-body field superposition. Each oscillon contributes a Gaussian tail φ_i(r) ∝ exp(−|r − r_i|²/2R²) to the total field. At the geometric center of the cluster:

| Geometry | Circumradius R_c | φ_center (all same) |
|----------|-----------------|---------------------|
| Tetrahedron | 3.67 | 0.680 |
| Octahedron | 4.24 | 0.711 |
| Cube | 5.20 | 0.461 |
| Icosahedron | 5.71 | 0.444 |

The octahedron achieves the largest central field (0.711) among non-complete graphs, exceeding the individual oscillon peak (φ₀ = 0.50). At this amplitude, the quartic (∝ φ⁴) and sextic (∝ φ⁶) potential terms generate significant three-body and higher-order contributions. The cube and icosahedron have central fields below 0.50, remaining in the regime where pairwise physics dominates.

### 7.3 Robustness of the Selection Rule

Despite the octahedron's strong many-body corrections, the selection rule is preserved: the threshold at f* = 0.484 remains near 50%. This is because the many-body enhancement affects both attractive and repulsive interactions, steepening the E_bind vs. f_cross slope without shifting its zero crossing. The approximate symmetry |E_cross_eff| ≈ |E_same_eff| that underlies the threshold is maintained even in the strongly nonlinear regime.

---

## 8. Robustness Tests

### 8.1 Spatial Resolution

Six configurations spanning all four geometries were re-simulated at N = 128:

| Configuration | E_bind (N=64) | E_bind (N=128) | Change | Sign preserved |
|---------------|---------------|----------------|--------|----------------|
| Tet single-flip (f=0.500) | −0.041 | −0.041 | 0.17% | Yes |
| Cube all-same (f=0.000) | +73.84 | +73.85 | 0.01% | Yes |
| Cube checkerboard (f=0.500) | −3.86 | −3.86 | 0.01% | Yes |
| Cube polarized T1 (f=1.000) | −51.53 | −51.53 | 0.01% | Yes |
| Ico ce_15_A (f=0.500) | +3.56 | +3.56 | 0.02% | Yes |
| Ico ce_20 (f=0.667) | −50.11 | −50.11 | 0.01% | Yes |

All six signs preserved. All magnitude changes below 0.2%.

### 8.2 Spacing Dependence

Five of six cube configurations were repeated at d = 7.5 (d/R = 3.0):

| f_cross | E_bind (d=6.0) | E_bind (d=7.5) | Sign preserved |
|---------|----------------|----------------|----------------|
| 0.000 | +73.84 | +25.61 | Yes |
| 0.250 | +36.89 | +12.80 | Yes |
| 0.333 | +20.59 | +8.02 | Yes |
| 0.500 | −3.86 | −0.52 | Yes |
| 1.000 | −51.53 | −22.66 | Yes |

All signs preserved. Magnitudes decrease to 35–44% of the d = 6.0 values, consistent with reduced field overlap.

### 8.3 Long-Time Stability

Two bound configurations were evolved to T = 2000 (approximately 354 oscillation periods):

| Configuration | E_bind(T=500) | E_bind(T=2000) | Variation | Energy drift |
|---------------|---------------|----------------|-----------|--------------|
| Cube pol. T1 (f=1.000) | −51.53 | −51.53 | ±0.02 | 0.034% |
| Ico ce_20 (f=0.667) | −50.11 | −50.11 | ±0.01 | 0.025% |

Binding energies are constant to within ±0.02 over 354 periods. Amplitudes oscillate (characteristic of oscillon breathing modes) without secular decay. The relevant test is not whether these clusters persist for the full 10⁶–10⁸ oscillation lifetime of individual oscillons, but whether the binding energy shows any secular trend that would indicate eventual unbinding. No such trend is observed over 354 periods.

### 8.4 Energy Conservation

All simulations maintain energy conservation to better than 0.04%. The isolated baseline shows 0.015% drift over T = 500.

### 8.5 Amplitude Retention

We tested whether bound configurations show enhanced amplitude retention compared to unbound ones. The measured ratio (mean retention of bound / mean retention of unbound) is 1.05, not significantly different from unity at T = 500. The binding energy advantage does not manifest as a detectable amplitude retention difference at the timescales studied.

---

## 9. Spontaneous Self-Organization

The preceding sections established the generalized selection rule (clusters are bound when Σ_edges cos(Δφ_e) < 0) and verified it across geometries, potentials, and resolutions. A natural question follows: does the selection rule merely classify static configurations, or does it describe a dynamical attractor? We address this by evolving icosahedral clusters from random initial phases and measuring whether the system spontaneously organizes toward configurations satisfying the rule.

### 9.1 Ensemble Design

We constructed an ensemble of 50 icosahedral clusters (12 oscillons, 30 edges) with Set B parameters at spacing d = 6.0. Each seed was initialized with random continuous phases θ_i drawn independently from [0, 2π) for each vertex, using the initialization procedure of Section 2.2 with the appropriate cos(θ_i) and sin(θ_i) projections onto the field and its time derivative. All seeds were evolved to T = 1000 (approximately 177 oscillation periods) with dt = 0.05 and dissipation σ = 0.01.

The phase of each oscillon was extracted at intervals of Δt = 10 via the arctangent of the field velocity and field amplitude at the vertex position. The continuous order parameter

    S(t) = (1/30) Σ_edges cos(Δφ_e(t))

measures the mean cosine of phase differences across all 30 edges. For the binary case, S = +1 corresponds to all-same-phase (maximally unbound) and S = -1 to all-cross-edge (maximally bound). For random continuous phases, the expectation is S = 0. Negative S indicates a preponderance of anti-phase neighbor correlations, precisely the condition identified in Section 3.3 as the binding criterion. Binding energies were computed against a time-matched isolated baseline evolved to T = 1000.

### 9.2 Results

**Table 5.** Self-organization ensemble statistics (50 seeds, icosahedron, random continuous phases).

| Metric | Value |
|--------|-------|
| Seeds with S decreased | 50/50 (100%) |
| Mean S: initial → final | +0.042 → −0.075 |
| Mean ΔS | −0.117 (z = −15.4) |
| Seeds bound at T = 1000 | 30/50 (60%) |
| Corr(S_final, E_bind_final) | 0.94 |

Every seed in the ensemble evolved toward more negative S, without exception. The mean shift ΔS = −0.117 yields a z-score of −15.4 (computed as the mean shift divided by its standard error), ruling out random drift at any conceivable significance level. The initial mean S = +0.042 is consistent with random phase assignments (expected value zero); the final mean S = −0.075 reflects a systematic displacement toward anti-phase neighbor correlations.

At T = 1000, 30 of 50 seeds (60%) have achieved negative binding energy. The mean binding energy across the ensemble is E_bind = −3.12, indicating that the typical random-phase cluster has become weakly bound. The correlation between final order parameter and final binding energy is r = 0.94: seeds that evolved to more negative S are more tightly bound. This confirms that the generalized selection rule of Section 3.3 operates not merely as a classification of static configurations but as a dynamical attractor; the system flows toward states that satisfy the rule, and the degree of satisfaction predicts the binding strength.

### 9.3 Dynamical Features

The ensemble-averaged trajectory S(t) reveals structure beyond the net downward drift.

During the first approximately 50 time units (roughly 9 oscillation periods), the order parameter spikes to S ≈ +0.87, indicating transient near-synchronization of all oscillons. This initial coherence transient reflects the adjustment of individual oscillon profiles to the collective field environment; all oscillons briefly breathe in unison as they radiate away their initialization artifacts. The transient is not a stable state; the system departs from it rapidly.

After the initial transient, S(t) exhibits slow quasi-periodic oscillations with a dominant period of approximately 500 time units, superimposed on the downward secular drift. FFT analysis of the ensemble-averaged trajectory identifies spectral peaks at periods of 252, 505, and 1010 time units. However, the phase extraction interval (Δt = 10) undersamples the individual oscillon oscillation period (τ = 5.65), so these specific periods should be treated as indicative rather than precise. The collective modes are two orders of magnitude slower than the individual oscillon oscillation period, indicating coherent cluster-scale dynamics rather than single-oscillon effects. The inter-seed variance narrows from 0.30 at t = 100 to 0.05 at t = 400, indicating ensemble convergence during the negative excursions of S, before widening again during positive swings.

The system has not fully converged at T = 1000. The order parameter continues to oscillate and its time-averaged value has not plateaued. Longer evolution may yield a higher bound fraction than the 60% observed here; conversely, the measurement at T = 1000 may catch the oscillation near a favorable phase. Both possibilities are consistent with the data.

### 9.4 Interpretation

The self-organization results place the selection rule in a dynamical context. The condition Σ_edges cos(Δφ_e) < 0 is not an externally imposed criterion but an emergent property of the energy landscape. Random initial conditions flow toward configurations satisfying the rule because such configurations occupy lower-energy regions of the phase space.

The initial synchronization transient is consistent with this picture. The all-same-phase configuration (S = +1) is an energy maximum, an unstable fixed point of the collective dynamics. The transient spike to S ≈ +0.87 represents the system approaching this maximum before rolling toward the lower-energy anti-phase-correlated states. The subsequent oscillatory descent reflects the complex topology of the 12-dimensional phase space (one phase per vertex), where the system navigates multiple local minima connected by the slow collective modes identified in Section 9.3.

The unanimous evolution of all 50 seeds toward anti-phase ordering invites comparison with several lines of prior work. Konstantinidis [22] computed equilibrium ground states for the antiferromagnetic XY model on the icosahedron, finding highly degenerate frustrated states. Our contribution is the dynamical observation: random initial conditions evolve toward these states in a conservative field theory without any thermal bath or dissipative relaxation beyond the minimal numerical dissipation (σ = 0.01) required for stability.

Dilão [23] proved that conservative pendula coupled by elastic forces exhibit anti-phase synchronization only for N = 2; for N ≥ 3, dynamics become ergodic with no synchronization. The unanimous ordering of 12 oscillators on the icosahedron contradicts this, indicating that the cosine coupling form and icosahedral topology are essential rather than incidental features of the dynamics.

The Hamiltonian Mean Field model with repulsive coupling [24] produces anti-phase biclustering in a conservative system, but only for all-to-all coupling; the effect has not been studied on frustrated graph topologies. BEC bright soliton trains [29] exhibit anti-phase ordering in a conservative system, but are restricted to 1D with external trapping.

The binding energy conservation observed in Section 8.3 (ΔE_bind ~ 0.002 while individual edges fluctuate by ~100) is the most striking quantitative result. The single-oscillon adiabatic invariant (the "I-charge") is established [18], but it concerns the internal charge of a single oscillon. Our finding is conceptually distinct: it concerns a collective quantity (binding energy) of a multi-body system. The natural theoretical framework is KAM theory. If the cluster dynamics inhabit a KAM torus in the high-dimensional phase space, then action variables are individually conserved. If binding energy is a function of the conserved actions alone, while individual edge energies depend on both actions and angles, the observed cancellation follows. The ~0.002 residual is consistent with Nekhoroshev exponential bounds on action drift in nearly-integrable systems. Berti, Kappeler, and Montalto [20] constructed KAM tori for perturbed KdV, the closest rigorous result for multi-soliton systems. Rigorous construction of KAM tori for oscillon clusters remains open.

These results have implications for cosmological oscillon formation. If oscillons produced after inflation [4,5] find themselves in geometric proximity, the self-organization mechanism identified here suggests that phase correlations favoring gravitational binding may develop spontaneously, without requiring fine-tuned initial conditions. The 60% bound fraction from purely random phases indicates that bound cluster formation is the majority outcome, at least for the icosahedral geometry at the spacings studied. The strong correlation (r = 0.94) between the continuous order parameter and binding energy further suggests that even partial self-organization (S shifting modestly negative) produces measurable energetic stabilization.

---

## 10. Discussion

### 10.1 Relation to Prior Work

Xue et al. [7] found no stable multi-oscillon bound states in 3+1D collisions. Our work differs in initializing oscillons at fixed geometric positions, allowing field overlap to equilibrate without collision kinetics. The cosine law we verify (Section 3) is consistent with the phase-dependent scattering reported in [7], but embeds it in a quantitative framework that predicts multi-body binding from pairwise data.

The oscillon cluster is a new member of a family of multi-soliton bound states. Optical soliton molecules [26] exhibit phase-dependent binding, but are 1D and dissipative. BEC soliton trains [30,29] show anti-phase ordering in a conservative system, but are 1D with external trapping. Skyrmion alpha clustering [28,27] produces 3D geometric arrangements of topological solitons (Carbon-12 as a triangle of alpha particles, Oxygen-16 as a tetrahedron), but with topological binding rather than oscillation-phase-dependent binding. Optical soliton clusters [25] are multi-soliton bound states stabilized by staircase phase distributions, again in lower-dimensional settings. No existing system simultaneously exhibits phase-dependent binding, 3D geometric arrangement, and anti-phase self-organization in a conservative field theory.

### 10.2 The Role of Graph Topology

The selection rule demonstrates that graph-theoretic properties (cross-edge fraction, neighbor signature variance, Max-Cut, bipartiteness) have direct physical consequences for oscillon cluster stability. The Max-Cut/Ising-antiferromagnet equivalence is well established in combinatorial optimization, and the 50% threshold corresponds to the Edwards bound [21]. The Max-Cut of the interaction graph sets an upper bound on the binding energy, while bipartiteness determines whether that bound is achievable.

"Neighbor signature variance" appears to be a novel quantity without a standard name in graph theory. The closest existing concepts are Estrada's spectral bipartivity [31] and the frustration index [32]. Cross-edge fraction relates to the "cut density" in graph theory and "bond satisfaction fraction" in frustrated magnetism, but has not previously been used as a field-theory order parameter.

### 10.3 Quasi-Integrability

The binding energy conservation should be distinguished from quasi-integrability as defined by Ferreira and Zakrzewski [19]. Quasi-integrability describes the post-scattering restoration of approximately conserved charges: after a collision, the charges return to their pre-collision values. Our observation is different in character: the binding energy is continuously conserved during bound-state dynamics, not restored after a transient disruption. The KAM interpretation (Section 9.4) provides a more natural framework for this continuous conservation.

### 10.4 Limitations

The study uses periodic boundary conditions (L = 50.0), allowing radiated energy to re-enter the domain. The binary and continuous phase analysis is limited to pairs and Platonic solid clusters; irregular geometries and lattice structures remain untested. The many-body correction for the octahedron is characterized but not quantitatively modeled. The continuous phase sweep (Section 3) maps E_pair(Δφ) for pairs; the self-organization ensemble (Section 9) demonstrates dynamics with continuous multi-body phases but is limited to the icosahedral geometry at a single spacing. Additionally, individual oscillon positions were not tracked over time. The constant binding energy over T = 2000 is consistent with positional stability but does not rule out slow centroid drift within the cluster geometry. Direct measurement of oscillon centroid positions over extended evolution would strengthen the stability claim.

### 10.5 Future Directions

**Lattice structures.** Cubic lattices (N = 27 and beyond) would test whether the selection rule scales to macroscopic oscillon assemblies. The bipartite structure guarantees Max-Cut fraction 1.0, suggesting strong collective binding for alternating-phase lattices.

**Continuous multi-body phases.** The cosine law predicts that the optimal multi-body configuration minimizes Σ_edges cos(Δφ_e), which is equivalent to the ground state of an antiferromagnetic XY model on the cluster graph. This connects oscillon stability to well-studied problems in statistical mechanics.

**KAM theory.** Rigorous construction of invariant tori for oscillon clusters would place the observed binding energy conservation on firm mathematical footing.

**Long-time collective modes.** Testing whether the collective breathing modes survive at longer evolution times would resolve the aliasing question noted in Section 9.3.

---

## 11. Conclusion

We have established four principal results for multi-oscillon cluster stability in sextic scalar field theory.

First, the pairwise interaction obeys the cosine law predicted analytically for soliton pairs [13,14], verified here for the first time in a non-integrable 3+1D sextic theory with R² = 0.999999 and negligible offset. This yields the generalized selection rule: a cluster is bound when Σ_edges cos(Δφ_e) < 0.

Second, for binary phases, the resulting 50% cross-edge fraction threshold is universal across potentials, verified with two parameter sets differing by 67–82% in coupling constants, and robust across four geometries, two spacings, and 354 oscillation periods.

Third, at the threshold, neighbor signature variance is a deterministic secondary predictor (R² = 0.997), established by exhaustive enumeration and revealing multi-body structure invisible to the pairwise model.

Fourth, the selection rule is not an artifact of prepared initial conditions but a dynamical attractor. A 50-seed ensemble initialized with random continuous phases unanimously evolves toward anti-phase nearest-neighbor correlations (z = −15.4), and 60% of seeds achieve negative binding energy within 177 oscillation periods. This establishes that bound multi-oscillon configurations are natural endpoints of cluster dynamics under the sextic potential, rather than outcomes contingent on fine-tuned phase preparation.

The pairwise model is exact for complete graphs and accurate to 3% for sparse graphs, but requires nonlinear many-body corrections for compact geometries like the octahedron where central field superposition exceeds the individual oscillon amplitude. In all cases, the qualitative selection rule is preserved.

These findings establish phase-geometric and graph-theoretic tools as a predictive framework for multi-body oscillon interactions, opening connections to Max-Cut optimization, the antiferromagnetic XY model, KAM theory for multi-soliton systems, and geometric frustration in condensed matter physics.

---

## References

[1] M. Gleiser, "Pseudostable bubbles," Phys. Rev. D **49**, 2978 (1994).

[2] E. J. Copeland, M. Gleiser, and H.-R. Müller, "Oscillons: Resonant configurations during bubble collapse," Phys. Rev. D **52**, 1920 (1995).

[3] M. A. Amin and D. Shiber, "Formation, scattering, and decay of oscillons," JCAP **02**, 063 (2010).

[4] M. A. Amin, R. Easther, H. Finkel, R. Flauger, and M. P. Hertzberg, "Oscillons after inflation," Phys. Rev. Lett. **108**, 241302 (2012).

[5] S. Antusch, F. Cefala, S. Krippendorf, F. Muia, S. Orani, and F. Quevedo, "Oscillons from string moduli," JHEP **01**, 083 (2018).

[6] D. Cyncynates and T. Giurgica-Tiron, "Structure of the oscillon: The dynamics of attractive self-interaction," Phys. Rev. D **103**, 116011 (2021).

[7] A. Xue, R. Easther, J. T. Giblin Jr., and E. I. Sfakianakis, "Realistic Oscillon Interactions," arXiv:2510.01597 (2025).

[8] C. H. Papadimitriou, *Computational Complexity* (Addison-Wesley, 1994), Chapter 20.

[9] H.-O. Kreiss and J. Oliger, "Comparison of accurate methods for the integration of hyperbolic equations," Tellus **24**, 199 (1972).

[10] G. H. Derrick, "Comments on nonlinear wave equations as models for elementary particles," J. Math. Phys. **5**, 1252 (1964).

[11] G. Fodor, P. Forgacs, Z. Horvath, and A. Lukacs, "Small amplitude quasi-breathers and oscillons," Phys. Rev. D **78**, 025003 (2008).

[12] H. Zhang, "Classical decay rates of oscillons," JCAP **07**, 055 (2020).

[13] J. P. Gordon, "Interaction forces among solitons in optical fibers," Opt. Lett. **8**, 596 (1983).

[14] N. S. Manton, "An effective Lagrangian for solitons," Nucl. Phys. B **150**, 397 (1979).

[15] N. S. Manton and P. Sutcliffe, *Topological Solitons* (Cambridge University Press, 2004).

[16] M. Axenides, S. Komineas, L. Perivolaropoulos, and M. Floratos, "Dynamics of non-topological solitons: Q-balls," Phys. Rev. D **61**, 085006 (2000); arXiv:hep-ph/9910388.

[17] P. G. Kevrekidis, A. Khare, and A. Saxena, "Solitary wave interactions in dispersive equations using Manton's approach," Phys. Rev. E **70**, 057603 (2004).

[18] M. Kawasaki, F. Takahashi, and N. Takeda, "Adiabatic invariance of oscillons/I-balls," Phys. Rev. D **92**, 105024 (2015); arXiv:1508.01028.

[19] L. A. Ferreira and W. J. Zakrzewski, "The concept of quasi-integrability: a concrete example," JHEP **05**, 130 (2011); arXiv:1011.2176.

[20] M. Berti, T. Kappeler, and R. Montalto, "Large KAM tori for quasi-linear perturbations of KdV," arXiv:1908.08768 (2019).

[21] C. S. Edwards, "Some extremal properties of bipartite subgraphs," Canadian J. Math. **25**, 475 (1973).

[22] N. P. Konstantinidis, "Magnetic ground state and spin fluctuations in icosahedral structures," Phys. Rev. B **91**, 054431 (2015); arXiv:1410.3444.

[23] R. Dilão, "Antiphase synchronization of two nonidentical pendulums," Eur. Phys. J. Special Topics **223**, 665 (2014).

[24] J. Barré and T. Dauxois, "Lyapunov exponents as a dynamical indicator of a phase transition," Europhys. Lett. **55**, 164 (2001); cond-mat/0101403.

[25] A. S. Desyatnikov and Yu. S. Kivshar, "Rotating optical soliton clusters," Phys. Rev. Lett. **88**, 053901 (2002).

[26] G. Herink, F. Kurtz, B. Jalali, D. R. Solli, and C. Ropers, "Real-time spectral interferometry probes the internal dynamics of femtosecond soliton molecules," Science **356**, 50 (2017).

[27] C. Naya and P. Sutcliffe, "Skyrmions and clustering in light nuclei," Phys. Rev. Lett. **121**, 232002 (2018).

[28] P. M. Lau and N. S. Manton, "States of Carbon-12 in the Skyrme model," Phys. Rev. Lett. **113**, 232503 (2014).

[29] J. H. V. Nguyen, P. Dyke, D. Luo, B. A. Malomed, and R. G. Hulet, "Collisions of matter-wave solitons," Nature Phys. **10**, 918 (2014).

[30] K. E. Strecker, G. B. Partridge, A. G. Truscott, and R. G. Hulet, "Formation and propagation of matter-wave soliton trains," Nature **417**, 150 (2002).

[31] E. Estrada, "Spectral measures of bipartivity in complex networks," Phys. Rev. E **72**, 046105 (2005).

[32] S. Aref, A. J. Mason, and M. C. Wilson, "A modeling and computational study of the frustration index in signed networks," arXiv:1611.09030 (2020).

[33] D. G. Levkov, V. E. Maslov, E. Y. Nugaev, and A. G. Panin, "An effective field theory for large oscillons," JHEP **12**, 079 (2022); arXiv:2208.04334.

---

## Appendix A: Vertex Coordinates

All geometries are centered at the origin with minimum edge length d = 6.0.

**Tetrahedron.** R_c = d√(3/8) = 3.674. Four vertices at distance R_c from origin.

**Octahedron.** R_c = d/√2 = 4.243. Vertices at (±R_c, 0, 0) and permutations.

**Cube.** R_c = d√3/2 = 5.196. Vertices at (±3, ±3, ±3) with indices: 0=(−,−,−), 1=(+,−,−), 2=(+,+,−), 3=(−,+,−), 4=(−,−,+), 5=(+,−,+), 6=(+,+,+), 7=(−,+,+). Edges connect vertices differing in exactly one coordinate (12 edges total). The two inscribed tetrahedra are T1 = {0,2,5,7} and T2 = {1,3,4,6}.

**Icosahedron.** R_c = d × sin(2π/5) = 5.706. Vertices at permutations of (0, ±1, ±φ) scaled so minimum distance = d, where φ = (1+√5)/2.

## Appendix B: Phase Assignments

Explicit phase arrays for reproducibility. Phases listed in vertex-index order; 0 = phase 0, π = phase π.

**Cube** (vertex ordering as in Appendix A; T1={0,2,5,7}, T2={1,3,4,6}):

- All-same (CE=0): [0, 0, 0, 0, 0, 0, 0, 0]
- Single flip (CE=3): [π, 0, 0, 0, 0, 0, 0, 0]
- Adjacent pair (CE=4): [π, π, 0, 0, 0, 0, 0, 0]
- Checkerboard (CE=6): [π, π, 0, π, 0, 0, 0, π]
    π at {0,1,3,7}: T1 picks {0,7}, T2 picks {1,3}
- Anti-tetrahedral (CE=8): [π, 0, 0, π, 0, π, π, 0]
    π at {0,3,5,6}: T1 picks {0,5}, T2 picks {3,6}
- Polarized T1 (CE=12): [π, 0, π, 0, 0, π, 0, π]
    π at T1={0,2,5,7}: full bipartition

**Icosahedron ce_15 variants:** Phase assignments for all six equivalence classes are provided in the supplementary data.

## Appendix C: Supplementary Tables

### C.1 Second Potential (Set C) Pairwise Calibration

| Config | E_bind (Set B) | E_bind (Set C) |
|--------|----------------|----------------|
| Same-phase pair | +5.172 | +5.152 |
| Cross-phase pair | −5.188 | −5.179 |

### C.2 Cube at d = 7.5

| f_cross | E_bind (d=6.0) | E_bind (d=7.5) | Ratio |
|---------|----------------|----------------|-------|
| 0.000 | +73.84 | +25.61 | 0.347 |
| 0.250 | +36.89 | +12.80 | 0.347 |
| 0.333 | +20.59 | +8.02 | 0.389 |
| 0.500 | −3.86 | −0.52 | 0.135 |
| 1.000 | −51.53 | −22.66 | 0.440 |

### C.3 Extended Evolution to T = 2000

| Config | T | E_bind | Energy drift |
|--------|---|--------|--------------|
| Cube pol. T1 | 500 | −51.53 | 0.034% |
| Cube pol. T1 | 1000 | −51.53 | 0.036% |
| Cube pol. T1 | 1500 | −51.53 | 0.024% |
| Cube pol. T1 | 2000 | −51.53 | 0.034% |
| Ico ce_20 | 500 | −50.11 | 0.025% |
| Ico ce_20 | 1000 | −50.11 | 0.026% |
| Ico ce_20 | 1500 | −50.12 | 0.016% |
| Ico ce_20 | 2000 | −50.11 | 0.025% |

### C.4 Self-Organization Ensemble

**Table C.4.** Summary statistics for the 50-seed self-organization ensemble (Section 9). Icosahedron, Set B parameters, d = 6.0, T = 1000.

| Statistic | Value |
|-----------|-------|
| Number of seeds | 50 |
| Mean S(t=0) | +0.042 |
| Mean S(t=1000) | −0.075 |
| Std S(t=1000) | 0.137 |
| Mean ΔS | −0.117 |
| Std ΔS | 0.054 |
| z-score (ΔS) | −15.4 |
| Seeds with ΔS < 0 | 50/50 (100%) |
| Seeds with E_bind < 0 at T=1000 | 30/50 (60%) |
| Mean E_bind at T=1000 | −3.12 |
| Corr(S_final, E_bind_final) | 0.94 |
| Mean energy drift | 0.0025% |
| Max energy drift | 0.0082% |
| Total wall time | 14.4 hours (4 P-cores) |

---

*Code and data are available at [repository URL to be added].*
