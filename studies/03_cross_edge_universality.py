"""
studies/03_cross_edge_universality.py
=====================================
Phase 3: Cross-Edge Universality Analysis

Loads Phase 1 (single oscillon baseline) and Phase 2 (cube + icosahedron)
results, performs cross-edge universality analysis, generates plots, and
saves a comprehensive report.
"""

import sys
import os
import json

import numpy as np

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
PHASE1_JSON = os.path.join(BASE_DIR, "outputs", "phase1", "set_B_baseline.json")
CUBE_SUMMARY = os.path.join(BASE_DIR, "outputs", "phase2", "cube", "summary.json")
ICO_SUMMARY  = os.path.join(BASE_DIR, "outputs", "phase2", "icosahedron", "summary.json")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs", "phase3")


def load_data():
    """Load all Phase 1 and Phase 2 results."""
    with open(PHASE1_JSON) as f:
        phase1 = json.load(f)
    with open(CUBE_SUMMARY) as f:
        cube = json.load(f)
    with open(ICO_SUMMARY) as f:
        ico = json.load(f)
    return phase1, cube, ico


def analyze_threshold(cube, ico):
    """Identify the stability transition point for each geometry."""
    results = {}
    for label, data, total_edges in [("cube", cube, 12), ("icosahedron", ico, 30)]:
        sorted_data = sorted(data, key=lambda x: x["cross_edge_fraction"])
        last_unstable = None
        first_stable = None
        for cfg in sorted_data:
            if cfg["verdict"] == "UNSTABLE":
                last_unstable = cfg
            elif cfg["verdict"] == "STABLE" and first_stable is None:
                first_stable = cfg
        results[label] = {
            "total_edges": total_edges,
            "last_unstable_frac": last_unstable["cross_edge_fraction"] if last_unstable else None,
            "last_unstable_name": last_unstable["config_name"] if last_unstable else None,
            "first_stable_frac": first_stable["cross_edge_fraction"] if first_stable else None,
            "first_stable_name": first_stable["config_name"] if first_stable else None,
        }
    return results


def compute_per_edge_contribution(data, label):
    """Compute per-edge energy contribution for a geometry."""
    sorted_data = sorted(data, key=lambda x: x["cross_edges"])
    e_bind_zero = sorted_data[0]["final_E_bind"]
    e_bind_max  = sorted_data[-1]["final_E_bind"]
    max_ce      = sorted_data[-1]["cross_edges"]
    delta_E     = e_bind_max - e_bind_zero
    per_edge    = delta_E / max_ce if max_ce > 0 else 0.0
    return {
        "geometry": label,
        "E_bind_0_cross": e_bind_zero,
        "E_bind_max_cross": e_bind_max,
        "max_cross_edges": max_ce,
        "delta_E": delta_E,
        "per_edge_contribution": per_edge,
    }


def compute_stabilization_ratios(phase1, cube, ico):
    """Compute stabilization ratio for each config vs isolated oscillon."""
    isolated_retention = phase1["amplitude_retention_final"]
    all_configs = []
    for geom_label, data in [("cube", cube), ("icosahedron", ico)]:
        for cfg in data:
            ratio = cfg["amplitude_retention"] / isolated_retention
            all_configs.append({
                "geometry": geom_label,
                "config_name": cfg["config_name"],
                "cross_edge_fraction": cfg["cross_edge_fraction"],
                "amplitude_retention": cfg["amplitude_retention"],
                "stabilization_ratio": ratio,
                "verdict": cfg["verdict"],
            })
    return all_configs, isolated_retention


def linear_fit(x, y):
    """Simple linear fit: y = slope * x + intercept."""
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def plot_ebind_vs_crossfrac(cube, ico, output_path):
    """Plot binding energy vs cross-edge fraction for both geometries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Cube data
    cx = [c["cross_edge_fraction"] for c in cube]
    cy = [c["final_E_bind"] for c in cube]
    ax.scatter(cx, cy, marker="s", c="steelblue", s=80, zorder=5,
               label="Cube (8 vertices, 12 edges)")
    # Label the new adjacent_flip point
    for c in cube:
        if c["config_name"] == "adjacent_flip":
            ax.annotate("adjacent_flip",
                        (c["cross_edge_fraction"], c["final_E_bind"]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color="steelblue", fontstyle="italic")

    # Icosahedron data
    ix = [c["cross_edge_fraction"] for c in ico]
    iy = [c["final_E_bind"] for c in ico]
    ax.scatter(ix, iy, marker="o", c="darkorange", s=80, zorder=5,
               label="Icosahedron (12 vertices, 30 edges)")

    # Linear fits
    cs, ci_val = linear_fit(cx, cy)
    x_fit = np.linspace(0, 0.75, 100)
    ax.plot(x_fit, cs * x_fit + ci_val, "--", color="steelblue", alpha=0.6,
            label="Cube fit (slope=%.1f)" % cs)

    isl, iint = linear_fit(ix, iy)
    ax.plot(x_fit, isl * x_fit + iint, "--", color="darkorange", alpha=0.6,
            label="Icosahedron fit (slope=%.1f)" % isl)

    # Reference lines
    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.5,
               label="E_bind = 0 (stability boundary)")
    ax.axvline(0.5, color="gray", linewidth=1, linestyle=":", alpha=0.7,
               label="50% cross-edge fraction")

    ax.set_xlabel("Cross-Edge Fraction", fontsize=12)
    ax.set_ylabel("E_bind(T=500)", fontsize=12)
    ax.set_title("Binding Energy vs Cross-Edge Fraction", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print("  Saved plot: %s" % output_path)


def plot_stabilization_ratio(stab_data, output_path):
    """Plot stabilization ratio vs cross-edge fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for geom, marker, color in [("cube", "s", "steelblue"),
                                 ("icosahedron", "o", "darkorange")]:
        subset = [s for s in stab_data if s["geometry"] == geom]
        x = [s["cross_edge_fraction"] for s in subset]
        y = [s["stabilization_ratio"] for s in subset]
        ax.scatter(x, y, marker=marker, c=color, s=80, zorder=5,
                   label=geom.capitalize())

    ax.axhline(1.0, color="black", linewidth=1, linestyle="-", alpha=0.5,
               label="Isolated oscillon baseline")
    ax.axvline(0.5, color="gray", linewidth=1, linestyle=":", alpha=0.7,
               label="50% cross-edge fraction")

    ax.set_xlabel("Cross-Edge Fraction", fontsize=12)
    ax.set_ylabel("Stabilization Ratio (retention / isolated retention)", fontsize=12)
    ax.set_title("Stabilization Ratio vs Cross-Edge Fraction", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print("  Saved plot: %s" % output_path)


def build_geometry_table(cube, ico, thresholds, per_edge_cube, per_edge_ico):
    """Build the geometry comparison table."""
    cube_sorted = sorted(cube, key=lambda x: x["final_E_bind"])
    ico_sorted  = sorted(ico, key=lambda x: x["final_E_bind"])
    return [
        {
            "geometry": "Cube",
            "graph_type": "Q3 (3-regular)",
            "vertices": 8,
            "edges": 12,
            "max_cross_frac": max(c["cross_edge_fraction"] for c in cube),
            "threshold_cross_frac": "%.3f - %.3f" % (
                thresholds["cube"]["last_unstable_frac"],
                thresholds["cube"]["first_stable_frac"]),
            "strongest_E_bind": cube_sorted[0]["final_E_bind"],
            "per_edge_contribution": per_edge_cube["per_edge_contribution"],
        },
        {
            "geometry": "Icosahedron",
            "graph_type": "Regular (5-regular)",
            "vertices": 12,
            "edges": 30,
            "max_cross_frac": max(c["cross_edge_fraction"] for c in ico),
            "threshold_cross_frac": "%.3f - %.3f" % (
                thresholds["icosahedron"]["last_unstable_frac"],
                thresholds["icosahedron"]["first_stable_frac"]),
            "strongest_E_bind": ico_sorted[0]["final_E_bind"],
            "per_edge_contribution": per_edge_ico["per_edge_contribution"],
        },
    ]


def write_report(thresholds, per_edge_cube, per_edge_ico, stab_data,
                 geom_table, isolated_retention, output_dir):
    """Write the comprehensive Phase 3 report."""
    lines = []
    lines.append("# Phase 3: Cross-Edge Universality Analysis Report")
    lines.append("")
    lines.append("## 1. Cross-Edge Threshold Identification")
    lines.append("")
    for label in ["cube", "icosahedron"]:
        t = thresholds[label]
        lines.append("### %s" % label.capitalize())
        lines.append("- Total edges: %d" % t["total_edges"])
        lines.append("- Last unstable config: %s (cross-edge fraction = %.3f)" % (
            t["last_unstable_name"], t["last_unstable_frac"]))
        lines.append("- First stable config: %s (cross-edge fraction = %.3f)" % (
            t["first_stable_name"], t["first_stable_frac"]))
        lines.append("- **Threshold lies between %.3f and %.3f**" % (
            t["last_unstable_frac"], t["first_stable_frac"]))
        lines.append("")
    lines.append("**Universal finding:** The stability threshold lies at or just above 50%")
    lines.append("cross-edge fraction for both geometries, with geometry-dependent refinement:")
    ct = thresholds["cube"]
    it = thresholds["icosahedron"]
    lines.append("- Cube: between %.3f (%s, unstable) and %.3f (%s, stable)" % (
        ct["last_unstable_frac"], ct["last_unstable_name"],
        ct["first_stable_frac"], ct["first_stable_name"]))
    lines.append("- Icosahedron: between %.3f (%s, unstable) and %.3f (%s, stable)" % (
        it["last_unstable_frac"], it["last_unstable_name"],
        it["first_stable_frac"], it["first_stable_name"]))
    lines.append("")

    lines.append("## 2. Binding Energy vs Cross-Edge Fraction")
    lines.append("")
    lines.append("![E_bind vs cross-edge fraction](ebind_vs_crossfrac.png)")
    lines.append("")
    lines.append("Both geometries show a clear monotonic decrease in binding energy")
    lines.append("with increasing cross-edge fraction. Negative E_bind indicates a")
    lines.append("bound (stable) multi-oscillon configuration.")
    lines.append("")

    lines.append("## 3. Per-Edge Energy Contribution")
    lines.append("")
    lines.append("| Geometry | E_bind(0 cross) | E_bind(max cross) | Max CE | delta_E | Per-Edge |")
    lines.append("|----------|-----------------|-------------------|--------|---------|----------|")
    for pe in [per_edge_cube, per_edge_ico]:
        lines.append("| %-12s | %+12.4f | %+12.4f | %6d | %+10.4f | %+8.4f |" % (
            pe["geometry"],
            pe["E_bind_0_cross"], pe["E_bind_max_cross"],
            pe["max_cross_edges"], pe["delta_E"], pe["per_edge_contribution"]))
    lines.append("")
    pe_ratio = per_edge_ico["per_edge_contribution"] / per_edge_cube["per_edge_contribution"]
    lines.append("Per-edge contribution ratio (icosahedron / cube): %.3f" % pe_ratio)
    lines.append("")

    lines.append("## 4. Stabilization Ratio (\"Unstable Alone, Stable Together\")")
    lines.append("")
    lines.append("Isolated oscillon amplitude retention at T=500: %.4f" % isolated_retention)
    lines.append("")
    lines.append("![Stabilization ratio](stabilization_ratio.png)")
    lines.append("")
    lines.append("| Geometry | Config | Cross-Frac | Amp Retention | Stab. Ratio | Verdict |")
    lines.append("|----------|--------|------------|---------------|-------------|---------|")
    for s in sorted(stab_data, key=lambda x: (x["geometry"], x["cross_edge_fraction"])):
        lines.append("| %-12s | %-15s | %10.3f | %13.4f | %11.4f | %-8s |" % (
            s["geometry"], s["config_name"], s["cross_edge_fraction"],
            s["amplitude_retention"], s["stabilization_ratio"], s["verdict"]))
    lines.append("")

    lines.append("## 5. Geometry Comparison Table")
    lines.append("")
    lines.append("| Geometry | Graph Type | V | E | Max X-Frac | Threshold X-Frac | Strongest E_bind | Per-Edge |")
    lines.append("|----------|------------|---|---|------------|------------------|------------------|----------|")
    for row in geom_table:
        lines.append("| %-12s | %-18s | %d | %2d | %.3f | %s | %+12.4f | %+8.4f |" % (
            row["geometry"], row["graph_type"], row["vertices"], row["edges"],
            row["max_cross_frac"], row["threshold_cross_frac"],
            row["strongest_E_bind"], row["per_edge_contribution"]))
    lines.append("")

    lines.append("## 6. Key Findings")
    lines.append("")
    lines.append("1. **Universal selection rule:** Multi-oscillon configurations become")
    lines.append("   stable (negative binding energy) when the cross-edge fraction exceeds")
    lines.append("   approximately 50%. This holds for both cube and icosahedron geometries.")
    lines.append("")
    lines.append("2. **Geometry-dependent threshold refinement:**")
    ct = thresholds["cube"]
    it = thresholds["icosahedron"]
    lines.append("   - Cube: stability transition between %.3f (%s) and %.3f (%s)" % (
        ct["last_unstable_frac"], ct["last_unstable_name"],
        ct["first_stable_frac"], ct["first_stable_name"]))
    lines.append("   - Icosahedron: stability transition between %.3f (%s) and %.3f (%s)" % (
        it["last_unstable_frac"], it["last_unstable_name"],
        it["first_stable_frac"], it["first_stable_name"]))
    lines.append("   - The threshold is at or just above 50% in both cases")
    lines.append("")
    lines.append("3. **Unstable alone, stable together:** Individual oscillons decay")
    lines.append("   (amplitude retention = %.4f at T=500), but multi-oscillon" % isolated_retention)
    lines.append("   configurations with sufficient cross-edges achieve negative binding")
    lines.append("   energy, demonstrating collective stabilization.")
    lines.append("")
    lines.append("4. **Per-edge energy contribution** differs between geometries:")
    lines.append("   cube per-edge = %.4f, icosahedron per-edge = %.4f (ratio = %.3f)." % (
        per_edge_cube["per_edge_contribution"],
        per_edge_ico["per_edge_contribution"], pe_ratio))
    lines.append("   The difference may reflect coordination-number dependence")
    lines.append("   (cube is 3-regular, icosahedron is 5-regular). Investigating")
    lines.append("   how per-edge contribution scales with vertex degree is a")
    lines.append("   natural direction for future work.")
    lines.append("")
    lines.append("5. **Amplitude retention caveat:** Amplitude retention does not correlate")
    lines.append("   cleanly with stability at T=500. Repulsive (unstable) configurations")
    lines.append("   can show high retention because oscillons are pushed apart and decay")
    lines.append("   independently across a larger volume. Binding energy remains the")
    lines.append("   reliable stability metric. Amplitude retention is included for")
    lines.append("   completeness but should not be interpreted as a direct measure of")
    lines.append("   stabilization.")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by studies/03_cross_edge_universality.py*")

    report_path = os.path.join(output_dir, "phase3_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  Saved report: %s" % report_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Phase 3: Cross-Edge Universality Analysis")
    print("Output dir: %s" % os.path.abspath(OUTPUT_DIR))
    print("")

    # Load data
    print("Loading Phase 1 and Phase 2 data...")
    phase1, cube, ico = load_data()
    print("  Phase 1 baseline: %d time points, retention=%.4f" % (
        len(phase1["t_series"]), phase1["amplitude_retention_final"]))
    print("  Cube configs: %d" % len(cube))
    print("  Icosahedron configs: %d" % len(ico))
    print("")

    # 1. Threshold identification
    print("1. Cross-edge threshold identification...")
    thresholds = analyze_threshold(cube, ico)
    for label in ["cube", "icosahedron"]:
        t = thresholds[label]
        print("   %s: threshold between %.3f (%s) and %.3f (%s)" % (
            label, t["last_unstable_frac"], t["last_unstable_name"],
            t["first_stable_frac"], t["first_stable_name"]))
    print("")

    # 2. Binding energy vs cross-edge fraction plot
    print("2. Plotting E_bind vs cross-edge fraction...")
    plot_ebind_vs_crossfrac(cube, ico,
                            os.path.join(OUTPUT_DIR, "ebind_vs_crossfrac.png"))
    print("")

    # 3. Per-edge energy contribution
    print("3. Per-edge energy contribution...")
    pe_cube = compute_per_edge_contribution(cube, "cube")
    pe_ico  = compute_per_edge_contribution(ico, "icosahedron")
    print("   %-14s  delta_E=%+10.4f  max_CE=%2d  per_edge=%+8.4f" % (
        "Cube:", pe_cube["delta_E"], pe_cube["max_cross_edges"],
        pe_cube["per_edge_contribution"]))
    print("   %-14s  delta_E=%+10.4f  max_CE=%2d  per_edge=%+8.4f" % (
        "Icosahedron:", pe_ico["delta_E"], pe_ico["max_cross_edges"],
        pe_ico["per_edge_contribution"]))
    ratio = pe_ico["per_edge_contribution"] / pe_cube["per_edge_contribution"]
    print("   Ratio (ico/cube): %.3f" % ratio)
    print("")

    # 4. Stabilization ratio
    print("4. Stabilization ratio analysis...")
    stab_data, isolated_ret = compute_stabilization_ratios(phase1, cube, ico)
    print("   Isolated retention: %.4f" % isolated_ret)
    plot_stabilization_ratio(stab_data,
                             os.path.join(OUTPUT_DIR, "stabilization_ratio.png"))
    print("")

    # 5. Geometry comparison table
    print("5. Geometry comparison table...")
    geom_table = build_geometry_table(cube, ico, thresholds, pe_cube, pe_ico)
    print("   %-14s  V=%2d  E=%2d  threshold=%.3f  strongest_E_bind=%+.4f  per_edge=%+.4f" % (
        "Cube", 8, 12, thresholds["cube"]["first_stable_frac"],
        geom_table[0]["strongest_E_bind"], geom_table[0]["per_edge_contribution"]))
    print("   %-14s  V=%2d  E=%2d  threshold=%.3f  strongest_E_bind=%+.4f  per_edge=%+.4f" % (
        "Icosahedron", 12, 30, thresholds["icosahedron"]["first_stable_frac"],
        geom_table[1]["strongest_E_bind"], geom_table[1]["per_edge_contribution"]))
    print("")

    # 6. Write report
    print("6. Writing comprehensive report...")
    write_report(thresholds, pe_cube, pe_ico, stab_data,
                 geom_table, isolated_ret, OUTPUT_DIR)
    print("")

    # Save all computed values as JSON
    phase3_data = {
        "thresholds": thresholds,
        "per_edge_cube": pe_cube,
        "per_edge_icosahedron": pe_ico,
        "per_edge_ratio_ico_over_cube": ratio,
        "isolated_retention": isolated_ret,
        "stabilization_ratios": stab_data,
        "geometry_table": geom_table,
    }
    data_path = os.path.join(OUTPUT_DIR, "phase3_data.json")
    with open(data_path, "w") as f:
        json.dump(phase3_data, f, indent=2)
    print("  Saved data: %s" % data_path)

    # Final summary
    print("")
    print("=" * 65)
    print("  PHASE 3 SUMMARY: CROSS-EDGE UNIVERSALITY")
    print("=" * 65)
    print("  Cube threshold:        %.3f -> %.3f" % (
        thresholds["cube"]["last_unstable_frac"],
        thresholds["cube"]["first_stable_frac"]))
    print("  Icosahedron threshold:  %.3f -> %.3f" % (
        thresholds["icosahedron"]["last_unstable_frac"],
        thresholds["icosahedron"]["first_stable_frac"]))
    print("  Universal threshold:    ~50% cross-edge fraction")
    print("  Per-edge (cube):        %+.4f" % pe_cube["per_edge_contribution"])
    print("  Per-edge (icosahedron): %+.4f" % pe_ico["per_edge_contribution"])
    print("  Per-edge ratio:         %.3f" % ratio)
    print("  Isolated retention:     %.4f" % isolated_ret)
    print("=" * 65)
    print("  Phase 3 complete. Output in: %s" % os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
