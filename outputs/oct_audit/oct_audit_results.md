# Octahedron Data Audit Results

Date: 2026-04-05
Parameters: Set B (phi0=0.5, R=2.5), N=64, L=50, dt=0.05, T=500, d=6.0

## Baseline

| Quantity | Value |
|----------|-------|
| E_single(T=0) | 1.346178e+01 |
| E_single(T=500) | 1.346381e+01 |
| Amplitude(T=500) | 0.099971 |

## Comparison Table

| Config | f_cross | E_bind (paper) | E_bind (PDE T=500) | E_bind (static) | PDE/static |
|--------|---------|-----------------|--------------------|-----------------|-----------:|
| all_same | 0.000 | 154.51 | 64.58 | 64.59 | 0.9999 |
| single_flip | 0.333 | 51.65 | 21.50 | 21.50 | 0.9999 |
| polarized | 0.500 | -32.01 | -2.90 | -2.90 | 0.9999 |
| balanced | 0.667 | -5.45 | -19.82 | -19.82 | 0.9999 |

## Derived Quantities

1. **Corrected oct threshold f*** = 0.4802 (old: 0.484)
2. **Corrected Ising R** = 1.9416 (old: 2.17), A = 25.307
3. **Iron peak oct entry**: E_bind/N = -0.4840 (old: -3.30)

## Many-Body Ratios

Using A_pair = 5.183 (cosine amplitude at d=6.0):

| Config | E_bind(PDE) | E_pair_sum | PDE/pairwise |
|--------|-------------|------------|-------------:|
| all_same | 64.58 | 62.20 | 1.0384 |
| single_flip | 21.50 | 20.73 | 1.0368 |
| polarized | -2.90 | 0.00 | inf |
| balanced | -19.82 | -20.73 | 0.9561 |

## Falsification Checks

All checks PASSED.
