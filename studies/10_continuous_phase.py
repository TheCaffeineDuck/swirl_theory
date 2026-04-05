#!/usr/bin/env python3
"""
studies/10_continuous_phase.py
==============================
Phase 10: Continuous Phase Pilot -- E_pair(Dphi)

Maps the full binding energy curve E_bind(Dphi) for relative phase
offsets from 0 to pi in increments of pi/6 (7 points).

Physics: An oscillon at phase offset Dphi has:
  phi(r, t=0)     = phi0 * cos(Dphi) * exp(-r^2 / 2R^2)
  phi_dot(r, t=0) = -omega_eff * phi0 * sin(Dphi) * exp(-r^2 / 2R^2)

The effective frequency omega_eff is NOT simply m. For a spatially
extended Gaussian mode, omega_eff = sqrt(k_eff^2 + m^2) where k_eff
depends on the spatial profile. We compute omega_eff numerically by
requiring that E(Dphi=0) = E(Dphi=pi/2), i.e., energy is independent
of the initial phase.

Validation: Isolated phase-shifted oscillons should have the same
energy regardless of Dphi. Verified before paired runs.

Execution order:
  1. Calibrate omega_eff from E(Dphi=0) -- blocking
  2. Validation: 3 isolated oscillons (Dphi = 0, pi/2, pi) -- blocking
  3. Pool(4): 7 paired runs at d=6.0
  4. Analysis -> continuous_phase_summary.json
"""

import os
import sys
import json
import time
import math
import multiprocessing as mp
from pathlib import Path

os.environ["NUMBA_NUM_THREADS"] = "1"

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

N_WORKERS = 4

# --- Parameters (Set B) ----------------------------------------------------

PARAMS = dict(
    N=64, L=50.0, m=1.0, g4=0.30, g6=0.055,
    phi0=0.5, R=2.5, dt=0.05, sigma=0.01, T_final=500.0, d=6.0,
)

# Known baseline from existing pairwise calibration
E_SINGLE_T500_REF = 13.46381472407115
E_BIND_SAME_REF = 5.172304543261493
E_BIND_CROSS_REF = -5.188490329426578

OUTDIR = Path("outputs/continuous_phase")

# Phase offsets to test
DPHIS = [0, math.pi / 6, math.pi / 3, math.pi / 2,
         2 * math.pi / 3, 5 * math.pi / 6, math.pi]
DPHI_DEGREES = [0, 30, 60, 90, 120, 150, 180]

# Will be set by calibration step
OMEGA_EFF = None


# --- Helpers ----------------------------------------------------------------

def atomic_write_json(data, filepath):
    """Write JSON atomically."""
    tmp = str(filepath) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(filepath))


def make_phase_shifted_ic(ev, center, phi0, R, dphi, omega):
    """Create a single phase-shifted oscillon IC.

    phi(r,0)     = phi0 * cos(dphi) * exp(-r^2 / 2R^2)
    phi_dot(r,0) = -omega * phi0 * sin(dphi) * exp(-r^2 / 2R^2)
    """
    import numpy as np
    dx = ev.X - center[0]
    dy = ev.Y - center[1]
    dz = ev.Z - center[2]
    r2 = dx**2 + dy**2 + dz**2
    profile = phi0 * np.exp(-r2 / (2.0 * R**2))
    phi = math.cos(dphi) * profile
    phi_dot = -omega * math.sin(dphi) * profile
    return phi, phi_dot


def make_pair_ic(ev, d, phi0, R, dphi, omega):
    """Create a pair of oscillons at (0,0,-d/2) and (0,0,+d/2).

    Oscillon 1: phase = 0 (standard)
    Oscillon 2: phase = dphi
    """
    import numpy as np
    center1 = [0.0, 0.0, -d / 2.0]
    center2 = [0.0, 0.0, +d / 2.0]

    phi1, phidot1 = make_phase_shifted_ic(ev, center1, phi0, R, 0.0, omega)
    phi2, phidot2 = make_phase_shifted_ic(ev, center2, phi0, R, dphi, omega)

    phi = phi1 + phi2
    phi_dot = phidot1 + phidot2
    return phi, phi_dot


# --- Calibration: compute omega_eff ----------------------------------------

def calibrate_omega_eff():
    """Compute omega_eff such that E(Dphi=0) = E(Dphi=pi/2).

    At Dphi=0: E = E_grad + E_pot (all in field, no velocity)
    At Dphi=pi/2 with omega: E = 0.5 * omega^2 * integral(phi0^2 * exp(-r^2/R^2) dV)

    We compute E(Dphi=0) and E_kin(Dphi=pi/2, omega=1), then:
    omega_eff = sqrt(E(0) / E_kin(pi/2, omega=1))
    """
    from engine.evolver import SexticEvolver
    import numpy as np

    print("=== CALIBRATION: Computing omega_eff ===")
    sys.stdout.flush()

    ev = SexticEvolver(
        PARAMS["N"], PARAMS["L"], PARAMS["m"],
        PARAMS["g4"], PARAMS["g6"],
        dissipation_sigma=PARAMS["sigma"],
    )

    # E at Dphi=0
    phi0, _ = make_phase_shifted_ic(
        ev, [0.0, 0.0, 0.0], PARAMS["phi0"], PARAMS["R"], 0.0, 1.0)
    ev.set_initial_conditions(phi0, np.zeros_like(phi0))
    E_dphi0 = ev.compute_energy()

    # E_kin at Dphi=pi/2 with omega=1 (all kinetic)
    _, phidot_pi2 = make_phase_shifted_ic(
        ev, [0.0, 0.0, 0.0], PARAMS["phi0"], PARAMS["R"], math.pi / 2, 1.0)
    E_kin_omega1 = 0.5 * np.sum(phidot_pi2**2) * ev.dx**3

    omega_eff = math.sqrt(E_dphi0 / E_kin_omega1)

    print("  E(Dphi=0)             = %.6e" % E_dphi0)
    print("  E_kin(Dphi=pi/2, w=1) = %.6e" % E_kin_omega1)
    print("  omega_eff = sqrt(E0/Ekin) = %.6f" % omega_eff)
    print("  (cf. m = %.1f, ratio omega_eff/m = %.4f)" % (PARAMS["m"], omega_eff / PARAMS["m"]))
    print()
    sys.stdout.flush()

    return omega_eff


# --- Validation: isolated phase-shifted oscillons --------------------------

def run_validation(omega_eff):
    """Run isolated oscillons at Dphi = 0, pi/2, pi to verify energy independence."""
    from engine.evolver import SexticEvolver
    import numpy as np

    print("=== VALIDATION: Isolated phase-shifted oscillons (omega_eff=%.4f) ===" % omega_eff)
    sys.stdout.flush()

    validation_dphis = [0.0, math.pi / 2, math.pi]
    validation_labels = ["0", "pi/2", "pi"]
    results = {}

    for dphi, label in zip(validation_dphis, validation_labels):
        ev = SexticEvolver(
            PARAMS["N"], PARAMS["L"], PARAMS["m"],
            PARAMS["g4"], PARAMS["g6"],
            dissipation_sigma=PARAMS["sigma"],
        )

        phi, phi_dot = make_phase_shifted_ic(
            ev, [0.0, 0.0, 0.0], PARAMS["phi0"], PARAMS["R"], dphi, omega_eff
        )
        ev.set_initial_conditions(phi, phi_dot)

        E0 = ev.compute_energy()
        amp0 = float(np.max(np.abs(ev.phi)))
        Ekin0 = 0.5 * np.sum(ev.phi_dot**2) * ev.dx**3

        results[label] = {
            "dphi_rad": dphi,
            "dphi_deg": math.degrees(dphi),
            "E_total_T0": E0,
            "max_amplitude_T0": amp0,
            "E_kinetic_T0": float(Ekin0),
            "cos_dphi": math.cos(dphi),
            "sin_dphi": math.sin(dphi),
        }

        print("  Dphi=%-5s: E_total(T=0) = %.6e  max|phi| = %.6e  E_kin = %.6e"
              % (label, E0, amp0, Ekin0))
        sys.stdout.flush()

    # Check energy independence
    energies = [v["E_total_T0"] for v in results.values()]
    E_mean = sum(energies) / len(energies)
    max_dev = max(abs(e - E_mean) / abs(E_mean) for e in energies) * 100
    passed = max_dev < 0.1

    # Check that Dphi=0 and Dphi=pi match the known baseline T=0
    E0_ref = 13.461782355144003
    E0_match_pct = abs(results["0"]["E_total_T0"] - E0_ref) / E0_ref * 100
    Epi_match_pct = abs(results["pi"]["E_total_T0"] - E0_ref) / E0_ref * 100
    baseline_match = E0_match_pct < 0.1 and Epi_match_pct < 0.1

    print()
    print("  Max deviation from mean: %.4f%%  %s" % (max_dev, "[PASS]" if passed else "[FAIL]"))
    print("  Dphi=0 vs baseline T=0 (%.6e): %.4f%%" % (E0_ref, E0_match_pct))
    print("  Dphi=pi vs baseline T=0: %.4f%%" % Epi_match_pct)
    print("  Baseline match: %s" % ("[PASS]" if baseline_match else "[FAIL]"))
    print()
    sys.stdout.flush()

    validation = {
        "results": results,
        "omega_eff": omega_eff,
        "E_mean": E_mean,
        "max_deviation_pct": max_dev,
        "energy_independent": passed,
        "baseline_match": baseline_match,
        "baseline_ref_T0": E0_ref,
    }

    outpath = OUTDIR / "validation_isolated.json"
    atomic_write_json(validation, outpath)
    print("  Saved: %s" % outpath)
    sys.stdout.flush()

    if not passed:
        print("  WARNING: Energy is NOT phase-independent (max_dev=%.4f%%)!" % max_dev)
        print("  This may indicate the nonlinear potential breaks the")
        print("  simple phase-rotation picture. Continuing with omega_eff=%.4f." % omega_eff)
        print()
        sys.stdout.flush()

    return validation


# --- Single paired config runner (for Pool) ---------------------------------

def run_paired_config(config):
    """Run one paired simulation with checkpointing."""
    from engine.evolver import SexticEvolver
    from engine.checkpoint import run_with_checkpointing
    import numpy as np

    name = config["name"]
    outpath = config["output_path"]
    omega = config["omega_eff"]

    # Cache check
    if os.path.exists(outpath):
        try:
            with open(outpath) as f:
                cached = json.load(f)
            if cached.get("completed", False):
                print("CACHED: %s" % name)
                sys.stdout.flush()
                return {
                    "completed": True,
                    "binding": cached["binding"],
                    "final_state": cached["final_state"],
                }
        except (json.JSONDecodeError, OSError):
            pass

    dphi = config["dphi"]
    ev = SexticEvolver(
        PARAMS["N"], PARAMS["L"], PARAMS["m"],
        PARAMS["g4"], PARAMS["g6"],
        dissipation_sigma=PARAMS["sigma"],
    )

    phi, phi_dot = make_pair_ic(
        ev, PARAMS["d"], PARAMS["phi0"], PARAMS["R"], dphi, omega
    )
    ev.set_initial_conditions(phi, phi_dot)

    ckpt_config = {
        "name": name,
        "params": {
            "N_grid": PARAMS["N"], "L": PARAMS["L"],
            "m": PARAMS["m"], "g4": PARAMS["g4"], "g6": PARAMS["g6"],
            "dt": PARAMS["dt"], "T_final": PARAMS["T_final"],
            "sigma_KO": PARAMS["sigma"],
        },
        "metadata": {
            "dphi_rad": dphi,
            "dphi_deg": config["dphi_deg"],
            "separation": PARAMS["d"],
            "omega_eff": omega,
        },
        "print_every": 2000,
        "record_every": 10,
    }

    result = run_with_checkpointing(ev, ckpt_config, outpath)

    # Compute binding energy
    E_total_final = result["final_state"]["E_total_final"]
    E_bind = E_total_final - 2 * E_SINGLE_T500_REF
    result["binding"] = {
        "E_single_T500": E_SINGLE_T500_REF,
        "n_oscillons": 2,
        "E_total_final": E_total_final,
        "E_bind": E_bind,
        "dphi_rad": dphi,
        "dphi_deg": config["dphi_deg"],
        "omega_eff": omega,
    }
    # Re-save with binding info
    atomic_write_json(result, outpath)
    # Return minimal data to avoid pipe serialization issues
    return {
        "completed": True,
        "binding": result["binding"],
        "final_state": result["final_state"],
    }


# --- Analysis ---------------------------------------------------------------

def run_analysis(paired_results, validation):
    """Compute cosine fit and print summary."""

    print()
    print("=" * 65)
    print("=== PHASE 10: CONTINUOUS PHASE RESULTS ===")
    print("=" * 65)
    print()

    omega_eff = validation["omega_eff"]

    # Validation summary
    print("VALIDATION (isolated oscillon energy independence):")
    val_res = validation["results"]
    for label in ["0", "pi/2", "pi"]:
        r = val_res[label]
        print("  Dphi=%-5s  E_single(T=0) = %.6e" % (label, r["E_total_T0"]))
    print("  Max deviation: %.4f%%  [%s]"
          % (validation["max_deviation_pct"],
             "PASS" if validation["energy_independent"] else "FAIL"))
    print("  omega_eff = %.6f (m = %.1f)" % (omega_eff, PARAMS["m"]))
    print()

    # Collect binding energies
    data_points = []
    for deg in DPHI_DEGREES:
        result = paired_results[deg]
        E_bind = result["binding"]["E_bind"]
        dphi_rad = result["binding"]["dphi_rad"]
        data_points.append({
            "dphi_deg": deg,
            "dphi_rad": dphi_rad,
            "E_bind": E_bind,
            "E_total_final": result["binding"]["E_total_final"],
        })

    # Cosine fit: E_bind = A * cos(dphi) + B
    n = len(data_points)
    cos_vals = [math.cos(p["dphi_rad"]) for p in data_points]
    E_vals = [p["E_bind"] for p in data_points]

    sum_cos2 = sum(c**2 for c in cos_vals)
    sum_cos = sum(cos_vals)
    sum_E_cos = sum(e * c for e, c in zip(E_vals, cos_vals))
    sum_E = sum(E_vals)

    det = sum_cos2 * n - sum_cos**2
    if abs(det) > 1e-15:
        A = (sum_E_cos * n - sum_E * sum_cos) / det
        B = (sum_cos2 * sum_E - sum_cos * sum_E_cos) / det
    else:
        A = 0.0
        B = sum_E / n

    # R^2
    E_mean = sum_E / n
    SS_tot = sum((e - E_mean)**2 for e in E_vals)
    SS_res = sum((e - (A * c + B))**2 for e, c in zip(E_vals, cos_vals))
    R2 = 1.0 - SS_res / SS_tot if SS_tot > 1e-15 else 0.0

    # Residuals
    residuals = [e - (A * c + B) for e, c in zip(E_vals, cos_vals)]

    # Print binding energy table
    print("PAIRWISE BINDING ENERGY vs PHASE DIFFERENCE:")
    print("  Dphi(deg)  Dphi(rad)   E_bind      Cosine_fit   Residual")
    print("  " + "-" * 60)
    for p, c, r in zip(data_points, cos_vals, residuals):
        fit_val = A * c + B
        print("  %-10d %-11.3f %-11.4f %-12.4f %-+.4f"
              % (p["dphi_deg"], p["dphi_rad"], p["E_bind"], fit_val, r))
    print()

    print("COSINE FIT: E_bind = A cos(Dphi) + B")
    print("  A = %.4f" % A)
    print("  B = %.4f" % B)
    print("  R^2 = %.6f" % R2)
    print()

    # Generalized threshold
    if abs(A) > 1e-10:
        threshold_ratio = -B / A
        print("GENERALIZED THRESHOLD:")
        print("  E_bind = 0 when cos(Dphi) = -B/A = %.4f" % threshold_ratio)
        if abs(threshold_ratio) <= 1:
            dphi_star = math.acos(threshold_ratio)
            print("  Dphi* = %.4f rad = %.1f deg" % (dphi_star, math.degrees(dphi_star)))
        else:
            dphi_star = None
            print("  (|B/A| > 1: no zero crossing in [0, pi])")
        binary_f_star = E_BIND_SAME_REF / (E_BIND_SAME_REF - E_BIND_CROSS_REF)
        print("  Binary (Dphi in {0, pi}): f* = %.4f" % binary_f_star)
        print("  Continuous: sum cos(Dphi_e) < %.4f * N_edges for binding"
              % threshold_ratio)
    else:
        threshold_ratio = None
        dphi_star = None
        binary_f_star = E_BIND_SAME_REF / (E_BIND_SAME_REF - E_BIND_CROSS_REF)
        print("  A ~ 0: no cosine dependence detected")
    print()

    # Deviations near pi/2
    dphi90_residual = None
    for p, r in zip(data_points, residuals):
        if p["dphi_deg"] == 90:
            dphi90_residual = r
            break
    if dphi90_residual is not None:
        print("NOTE: Dphi=90 residual = %.4f" % dphi90_residual)
        if abs(dphi90_residual) > 0.5:
            print("  -> Significant deviation from cosine fit at this point")
        else:
            print("  -> Consistent with cosine fit")
    print()

    # Consistency check with known endpoints
    print("ENDPOINT CONSISTENCY:")
    dphi0_ebind = data_points[0]["E_bind"]
    dphi180_ebind = data_points[-1]["E_bind"]
    print("  E_bind(Dphi=0)   = %.4f  (ref: %.4f, diff: %.4f)"
          % (dphi0_ebind, E_BIND_SAME_REF,
             dphi0_ebind - E_BIND_SAME_REF))
    print("  E_bind(Dphi=pi)  = %.4f  (ref: %.4f, diff: %.4f)"
          % (dphi180_ebind, E_BIND_CROSS_REF,
             dphi180_ebind - E_BIND_CROSS_REF))
    print()

    # Build summary JSON
    summary = {
        "description": "Phase 10: Continuous Phase Pilot -- E_pair(Dphi)",
        "parameters": {
            "N": PARAMS["N"], "L": PARAMS["L"],
            "m": PARAMS["m"], "g4": PARAMS["g4"], "g6": PARAMS["g6"],
            "phi0": PARAMS["phi0"], "R": PARAMS["R"],
            "dt": PARAMS["dt"], "sigma": PARAMS["sigma"],
            "T_final": PARAMS["T_final"], "d": PARAMS["d"],
            "omega_eff": omega_eff,
        },
        "E_single_T500": E_SINGLE_T500_REF,
        "validation": {
            "E_single_T0": {label: val_res[label]["E_total_T0"]
                            for label in ["0", "pi/2", "pi"]},
            "max_deviation_pct": validation["max_deviation_pct"],
            "energy_independent": validation["energy_independent"],
            "baseline_match": validation["baseline_match"],
            "omega_eff": omega_eff,
        },
        "data_points": data_points,
        "cosine_fit": {
            "A": A,
            "B": B,
            "R_squared": R2,
            "residuals": {str(p["dphi_deg"]): r
                          for p, r in zip(data_points, residuals)},
        },
        "generalized_threshold": {
            "neg_B_over_A": threshold_ratio,
            "dphi_star_rad": dphi_star,
            "dphi_star_deg": math.degrees(dphi_star) if dphi_star else None,
            "binary_f_star": binary_f_star,
        },
        "endpoint_consistency": {
            "E_bind_dphi0": dphi0_ebind,
            "E_bind_dphi0_ref": E_BIND_SAME_REF,
            "E_bind_dphi180": dphi180_ebind,
            "E_bind_dphi180_ref": E_BIND_CROSS_REF,
        },
        "generalization_statement": (
            "For a multi-oscillon cluster with continuous phases {phi_i}, "
            "the pairwise binding energy is: "
            "E_bind = sum_edges [A cos(phi_i - phi_j) + B]. "
            "The cluster is bound when sum_edges cos(Dphi_e) < -B*N_edges/A. "
            "For binary phases (Dphi in {0, pi}), this reduces to the "
            "50%% cross-edge fraction rule."
        ) if R2 > 0.99 else (
            "Cosine fit R^2 = %.4f < 0.99. The binding energy has "
            "significant non-cosine contributions." % R2
        ),
        "completed": True,
    }

    outpath = OUTDIR / "continuous_phase_summary.json"
    atomic_write_json(summary, outpath)
    print("Saved: %s" % outpath)

    return summary


# --- Main -------------------------------------------------------------------

def main():
    wall_start = time.perf_counter()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Calibrate omega_eff
    omega_eff = calibrate_omega_eff()

    # Step 2: Validation
    validation = run_validation(omega_eff)

    # Step 3: Build paired configs
    configs = []
    for dphi, deg in zip(DPHIS, DPHI_DEGREES):
        outpath = str(OUTDIR / ("pair_dphi_%d.json" % deg))
        configs.append({
            "name": "pair_dphi_%d" % deg,
            "dphi": dphi,
            "dphi_deg": deg,
            "output_path": outpath,
            "omega_eff": omega_eff,
        })

    # Step 4: Run all 7 paired simulations with Pool(4)
    print("Running %d paired configs on %d workers..." % (len(configs), N_WORKERS))
    sys.stdout.flush()

    with mp.Pool(N_WORKERS) as pool:
        results_list = pool.map(run_paired_config, configs)

    # Collect results by degree
    paired_results = {}
    for config, result in zip(configs, results_list):
        paired_results[config["dphi_deg"]] = result

    # Step 5: Analysis
    summary = run_analysis(paired_results, validation)

    # Cleanup
    from engine.checkpoint import cleanup_study
    cleanup_study(OUTDIR)

    wall_total = time.perf_counter() - wall_start
    print()
    print("Total wall time: %.1f min" % (wall_total / 60.0))


if __name__ == "__main__":
    main()
