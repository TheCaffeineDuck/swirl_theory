# Consolidated Literature Review: Final Novelty Assessment
**Date:** 2026-03-22  
**Sources:** Initial web search sweep + Deep Research gap analysis  
**Purpose:** Definitive pre-submission novelty map for paper v7→v8

---

## CRITICAL ACTION ITEM: Reframe the cosine law

**The cosine pairwise interaction is NOT new.** Gordon (1983) derived it analytically for NLS solitons. Manton (1979) established the tail-overlap mechanism. Kevrekidis, Khare & Saxena (PRE 2004) extended it to sine-Gordon breathers. Axenides et al. (PRD 2000, hep-ph/9910388) showed consistency for Q-balls. This is 40-year-old textbook material (Manton & Sutcliffe, *Topological Solitons*, Cambridge 2004).

**What you CAN claim:** First explicit numerical verification in a non-integrable 3+1D sextic potential. The R²=0.999999 demonstrates that higher-order corrections are negligible — confirming universality of the Gordon/Manton mechanism beyond integrable systems.

**Paper action:** Add Gordon (1983), Manton (1979), Axenides et al. (2000) to references. Rewrite any language claiming discovery of the cosine law as "verification and extension." This is urgent — presenting this as a discovery would be the single most damaging error in the paper.

---

## Novelty Ranking (strongest → weakest)

### 1. BINDING ENERGY AS COLLECTIVE ADIABATIC INVARIANT — GENUINELY NOVEL
**Confidence: VERY HIGH that this is new**

The specific finding: 30 edge energies each fluctuate by ~100 units; their sum (binding energy) is conserved to 0.002. A ~50,000× cancellation ratio maintained continuously during large phase rearrangements.

No precedent exists anywhere in soliton physics, despite three adjacent frameworks:
- Single-oscillon adiabatic invariants (Kawasaki et al. 2015) treat individual objects only
- Quasi-integrability (Ferreira & Zakrzewski, JHEP 2011) gives post-scattering restoration, not continuous conservation during bound-state dynamics
- KAM/Nekhoroshev theory provides the natural framework but no one has constructed KAM tori for oscillon clusters or identified binding energy as a KAM-protected quantity

**Theoretical framing opportunity:** If binding energy is a function of conserved KAM action variables alone (while individual edge energies depend on both actions and angles), your observation follows naturally. The ~0.002 residual could correspond to Nekhoroshev exponential bounds on action drift. Proposing this interpretation would itself be a theoretical contribution.

**Key references to cite:**
- Kawasaki, Takahashi & Takeda, PRD 92, 105024 (2015) [single-oscillon adiabatic invariant]
- Ferreira & Zakrzewski, JHEP 05, 130 (2011) [quasi-integrability]
- Berti, Kappeler & Montalto, arXiv:1908.08768 (2019) [KAM tori for perturbed KdV]

---

### 2. GRAPH-THEORETIC SELECTION RULES — NOVEL APPLICATION + NOVEL QUANTITY
**Confidence: VERY HIGH that this is new**

Max-Cut ↔ Ising antiferromagnet equivalence is established (textbook). But:
- Max-Cut has **never** been applied as a stability constraint in classical field theory
- The 50% threshold corresponds precisely to the **Edwards bound** (1973): maxcut ≥ m/2 + (n-1)/4. Configurations below random-cut performance are unstable. This is a clean mathematical correspondence you should highlight
- **"Neighbor signature variance" has no standard name** in graph theory. The closest existing concepts are Estrada's spectral bipartivity (PRE 2005) and local balance index (2024), but the variance of cross-edge counts per vertex is novel
- Cross-edge fraction as a physics quantity is related to "cut density" in graph theory and "bond satisfaction fraction" in frustrated magnetism, but has never been used as a field-theory order parameter

**Key references to cite:**
- Edwards, Can. J. Math. 25, 475 (1973) [the m/2 bound]
- Aref, Mason & Wilson, arXiv:1611.09030 (2020) [frustration index = |E| - maxcut]
- Estrada, PRE 72, 046105 (2005) [spectral bipartivity — related but distinct]

---

### 3. UNANIMOUS ANTI-PHASE SELF-ORGANIZATION IN A CONSERVATIVE SYSTEM — NOVEL
**Confidence: HIGH that this is new**

Sits in a precise gap between three mature communities:
- **XY antiferromagnet on icosahedron:** Konstantinidis (arXiv:2511.06004, 2025; arXiv:1410.3444, 2014) computed equilibrium ground states — 72 degenerate states, highly frustrated. But these are Monte Carlo equilibrium calculations, NOT Hamiltonian dynamics
- **Hamiltonian Mean Field (HMF) model:** Barré & Dauxois (2001) found anti-phase biclustering with repulsive coupling — but only for all-to-all coupling (complete graph), not frustrated topology
- **Critical negative result:** Dilão (Eur. Phys. J. Special Topics 2014) showed conservative coupled pendula exhibit anti-phase synchronization ONLY for N=2; for N≥3, dynamics become ergodic. Your unanimous ordering of 12 oscillators contradicts this — suggesting the specific cosine coupling + icosahedral topology is essential

**BEC bright soliton trains** (Strecker et al., Nature 2002; Nguyen et al., Nature Physics 2014) show anti-phase ordering in a conservative system — but in 1D with external trapping. Your result is 3D, untrapped, on a frustrated graph.

**Key references to cite:**
- Konstantinidis, arXiv:1410.3444 (2014) [icosahedral AF ground state]
- Barré & Dauxois, cond-mat/0101403 (2001) [repulsive HMF biclustering]
- Dilão, Eur. Phys. J. Special Topics 223, 665 (2014) [N≥3 ergodicity — your result contradicts this]
- Nguyen et al., Nature Physics 10, 918 (2014) [BEC anti-phase soliton trains]

---

### 4. MULTI-OSCILLON CLUSTERS AS A NEW CLASS OF BOUND STATE — NOVEL SYNTHESIS
**Confidence: HIGH**

No existing system simultaneously exhibits (i) phase-dependent binding, (ii) 3D geometric arrangement, and (iii) anti-phase self-organization. Three systems each have two of three:
- **Optical soliton molecules** (Herink et al., Science 2017): phase-dependent binding + anti-phase preference, but 1D dissipative
- **BEC soliton trains** (Strecker et al. 2002): anti-phase ordering + conservative, but 1D with trapping
- **Skyrmion alpha clustering** (Naya & Sutcliffe, PRL 2018; Lau & Manton, PRL 2014): 3D geometric arrangement, but topological binding, not phase-dependent

**The Skyrmion/alpha-cluster analogy is the most compelling unexploited connection.** Carbon-12 as a triangle of alpha particles, Oxygen-16 as a tetrahedron — structurally parallel to your Platonic solid oscillon clusters but with fundamentally different binding physics.

**Lozanov, Sasaki & Takhistov (arXiv:2309.14193):** "Clustered" refers to cosmological-scale gravitational aggregation, NOT few-body bound states. Not relevant to your work.

---

### 5. COLLECTIVE BREATHING MODES — NOVEL OBSERVATION, EXPECTED PHYSICS
**Confidence: HIGH for novelty of specific observation**

The physics is qualitatively expected (timescale separation is generic in coupled oscillator systems) but has never been documented for finite oscillon clusters. Quantitative parallels:
- **Optical soliton crystal breathers** (Weng et al., Light: Sci. Appl. 2024): timescale ratio ~500× (dissipative)
- **Tkachenko modes of BEC vortex lattices** (Baym, PRL 2003): 100-1000× separation (conservative — best analogue)
- **Skyrmion crystal breathing** (Mochizuki, PRL 2012; Kim et al., J. Appl. Phys. 2018): modest 1-10× ratio

**CAVEAT:** Your Phase 13 identified that breathing periods (252, 505, 1010 time units) may be aliasing artifacts from Δt=10 undersampling τ=5.65. Resolve this before claiming specific periods.

---

### 6. COSINE PAIRWISE LAW — NOT NOVEL (see critical action item above)
**Confidence: LOW for novelty. HIGH that it's a useful verification.**

Reframe as: "We verify that the analytically predicted cosine interaction law (Gordon 1983) holds quantitatively for oscillons in non-integrable 3+1D sextic scalar field theory, with R²=0.999999 demonstrating negligible higher-order corrections at the inter-oscillon separations studied."

---

## Complete Reference List to Add

### Must cite (missing from current paper):
| Reference | Why |
|-----------|-----|
| Gordon, Opt. Lett. 8, 596 (1983) | Derived the cosine interaction law |
| Manton, Nucl. Phys. B 150, 397 (1979) | Established tail-overlap soliton interaction mechanism |
| Manton & Sutcliffe, *Topological Solitons* (Cambridge, 2004) | Textbook treatment of cosine law |
| Axenides et al., PRD 61, 085006 (2000); hep-ph/9910388 | Cosine interaction consistency for Q-balls |
| Edwards, Can. J. Math. 25, 475 (1973) | The m/2 bound on Max-Cut (your 50% threshold) |
| Kawasaki et al., PRD 92, 105024 (2015); arXiv:1508.01028 | Single-oscillon adiabatic invariant |

### Should cite (strengthen novelty framing):
| Reference | Why |
|-----------|-----|
| Konstantinidis, arXiv:1410.3444 (2014) | AF XY model on icosahedron (equilibrium) |
| Ferreira & Zakrzewski, JHEP 05, 130 (2011) | Quasi-integrability (contrast with your finding) |
| Dilão, EPJST 223, 665 (2014) | N≥3 conservative anti-phase ordering fails (you contradict) |
| Barré & Dauxois, cond-mat/0101403 (2001) | Repulsive HMF biclustering |
| Naya & Sutcliffe, PRL 121, 232002 (2018) | Skyrmion alpha clustering (structural analogue) |
| Desyatnikov & Kivshar, PRL 88, 053901 (2002) | Optical soliton clusters |
| Estrada, PRE 72, 046105 (2005) | Spectral bipartivity (related to your variance) |
| Aref et al., arXiv:1611.09030 (2020) | Frustration index definition |

### Consider citing (enriches discussion):
| Reference | Why |
|-----------|-----|
| Nguyen et al., Nature Physics 10, 918 (2014) | BEC anti-phase soliton trains |
| Baym, PRL 91, 110402 (2003) | Tkachenko modes (timescale separation analogue) |
| Kevrekidis, Khare & Saxena, PRE 70, 057603 (2004) | Cosine law for sine-Gordon breathers |
| Herink et al., Science 356, 50 (2017) | Optical soliton molecule phase dynamics |
| Berti, Kappeler & Montalto, arXiv:1908.08768 (2019) | KAM tori for perturbed KdV |
| Kim et al., npj Quantum Materials (2023) | Frustrated Ising on icosahedron |

---

## Paper v8 Rewrite Priorities

1. **URGENT:** Rewrite cosine law language throughout. Every instance of "we discover" or "we find" regarding E_pair = A·cos(Δφ) + B must become "we verify" or "we confirm." Add Gordon (1983) citation.

2. **Section 1 (Introduction):** Add paragraph on established cosine interaction law with citations. Position your work as extending from pairwise verification to multi-body selection rules.

3. **Section on binding energy conservation:** Frame via KAM/Nekhoroshev. This is your strongest novel result — give it prominent treatment. Cite Kawasaki (2015) for contrast with single-oscillon case.

4. **Section on self-organization:** Cite Konstantinidis (2014) for icosahedral AF equilibrium, note your contribution is the DYNAMICAL observation. Cite Dilão (2014) as the N≥3 negative result you contradict.

5. **Section on graph theory:** Acknowledge Edwards bound (1973) as the mathematical foundation for your 50% threshold. Note that neighbor signature variance appears to be a novel quantity.

6. **Discussion:** Draw the Skyrmion/alpha-cluster analogy (Naya & Sutcliffe 2018). Position oscillon clusters as a new member of the "geometric multi-soliton bound state" family alongside optical soliton molecules, BEC soliton trains, and nuclear alpha clusters — unified by phase-dependent binding but unique in combining 3D geometry with conservative anti-phase ordering.
