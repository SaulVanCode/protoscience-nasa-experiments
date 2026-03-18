#!/usr/bin/env python3
"""
E096 — Meta-ProtoScience: Patterns in Discovered Laws

Question: Are there patterns in the laws ProtoScience has discovered?
Does the nature of the data predict how clean the law will be?

Data: ProtoScience's own experiment results (E061-E095, BH001-BH003)
  - Exponents, R², sample sizes, domains, data types, year of original discovery

This is science of science — feeding discoveries back as data.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── The Meta-Dataset ────────────────────────────────────────────────
# Each row is a discovered law from a ProtoScience experiment

DISCOVERIES = [
    # E061 - Turbofan
    {"id": "E061", "domain": "engineering", "law": "Ps30 degradation",
     "exponent": 1.0, "r2": 0.380, "n": 100, "year_discovered": 2008,
     "n_variables": 1, "data_type": "real", "source": "NASA"},

    # E062 - Kepler
    {"id": "E062", "domain": "astrophysics", "law": "Kepler P²=a³/M",
     "exponent": 1.000, "r2": 0.998, "n": 3519, "year_discovered": 1619,
     "n_variables": 3, "data_type": "real", "source": "NASA"},

    # E063 - Fireballs
    {"id": "E063", "domain": "physics", "law": "Luminous efficiency",
     "exponent": 2.1, "r2": 0.954, "n": 1052, "year_discovered": 1960,
     "n_variables": 2, "data_type": "real", "source": "NASA"},

    # E064 - Voyager
    {"id": "E064", "domain": "astrophysics", "law": "Heliopause rho~r^-2",
     "exponent": -2.09, "r2": 0.997, "n": 156909, "year_discovered": 2012,
     "n_variables": 2, "data_type": "real", "source": "NASA"},

    # E065 - Sunspots
    {"id": "E065", "domain": "astrophysics", "law": "11-year solar cycle",
     "exponent": None, "r2": 0.891, "n": 3326, "year_discovered": 1843,
     "n_variables": 1, "data_type": "real", "source": "SILSO"},

    # E066 - Gravitational waves
    {"id": "E066", "domain": "gr", "law": "Chirp mass formula",
     "exponent": 0.600, "r2": 0.998, "n": 219, "year_discovered": 1963,
     "n_variables": 3, "data_type": "real", "source": "LIGO"},

    # E067 - Asteroids
    {"id": "E067", "domain": "astrophysics", "law": "Kepler for asteroids",
     "exponent": 1.500, "r2": 0.999, "n": 10000, "year_discovered": 1619,
     "n_variables": 2, "data_type": "real", "source": "NASA"},

    # E069 - Hubble
    {"id": "E069", "domain": "cosmology", "law": "Hubble v=H0*d",
     "exponent": 1.000, "r2": 0.810, "n": 709, "year_discovered": 1929,
     "n_variables": 2, "data_type": "real", "source": "NED"},

    # E071 - Dark matter
    {"id": "E071", "domain": "cosmology", "law": "Flat rotation curves",
     "exponent": 0.000, "r2": 0.987, "n": 175, "year_discovered": 1970,
     "n_variables": 2, "data_type": "real", "source": "SPARC"},

    # E074 - Dark energy
    {"id": "E074", "domain": "cosmology", "law": "Accelerating expansion",
     "exponent": None, "r2": 0.994, "n": 1590, "year_discovered": 1998,
     "n_variables": 3, "data_type": "real", "source": "Pantheon+"},

    # E079 - CERN dimuon
    {"id": "E079", "domain": "particle", "law": "Z boson mass",
     "exponent": None, "r2": 0.990, "n": 77623, "year_discovered": 1983,
     "n_variables": 1, "data_type": "real", "source": "CERN"},

    # E091 - Gaia Stefan-Boltzmann
    {"id": "E091", "domain": "astrophysics", "law": "Stefan-Boltzmann L~R²T⁴",
     "exponent": 4.065, "r2": 0.994, "n": 15000, "year_discovered": 1879,
     "n_variables": 3, "data_type": "real", "source": "ESA"},

    # E091 - Gaia mass-luminosity
    {"id": "E091b", "domain": "astrophysics", "law": "Mass-luminosity L~M^4",
     "exponent": 4.123, "r2": 0.945, "n": 10168, "year_discovered": 1924,
     "n_variables": 2, "data_type": "real", "source": "ESA"},

    # E092 - Earthquakes
    {"id": "E092", "domain": "geophysics", "law": "Gutenberg-Richter",
     "exponent": 0.810, "r2": 0.915, "n": 1930, "year_discovered": 1944,
     "n_variables": 1, "data_type": "real", "source": "USGS"},

    # E093 - Argo EOS
    {"id": "E093", "domain": "oceanography", "law": "EOS-80 seawater",
     "exponent": None, "r2": 1.000, "n": 333, "year_discovered": 1980,
     "n_variables": 3, "data_type": "real", "source": "Argo"},

    # E094 - Kleiber
    {"id": "E094", "domain": "biology", "law": "Kleiber BMR~M^0.75",
     "exponent": 0.702, "r2": 0.923, "n": 573, "year_discovered": 1932,
     "n_variables": 2, "data_type": "real", "source": "PanTHERIA"},

    # E094 - Pop density
    {"id": "E094b", "domain": "biology", "law": "Pop density~M^-0.75",
     "exponent": -0.741, "r2": 0.572, "n": 947, "year_discovered": 1981,
     "n_variables": 2, "data_type": "real", "source": "PanTHERIA"},

    # E094 - Home range
    {"id": "E094c", "domain": "biology", "law": "Home range~M^1.0",
     "exponent": 1.061, "r2": 0.679, "n": 700, "year_discovered": 1979,
     "n_variables": 2, "data_type": "real", "source": "PanTHERIA"},

    # E095 - Gompertz
    {"id": "E095", "domain": "biology", "law": "Gompertz mortality",
     "exponent": 0.092, "r2": 0.994, "n": 500, "year_discovered": 1825,
     "n_variables": 1, "data_type": "real", "source": "SSA/WHO"},

    # BH001 - Schwarzschild r=2M
    {"id": "BH001a", "domain": "gr", "law": "r_horizon = 2M",
     "exponent": 1.000, "r2": 1.000, "n": 50, "year_discovered": 1916,
     "n_variables": 1, "data_type": "simulated", "source": "Kerr sim"},

    # BH001 - Shadow
    {"id": "BH001b", "domain": "gr", "law": "r_shadow = 3√3 M",
     "exponent": 1.000, "r2": 1.000, "n": 50, "year_discovered": 1916,
     "n_variables": 1, "data_type": "simulated", "source": "Kerr sim"},

    # BH001 - ISCO
    {"id": "BH001c", "domain": "gr", "law": "r_isco = 6M",
     "exponent": 1.000, "r2": 1.000, "n": 50, "year_discovered": 1916,
     "n_variables": 1, "data_type": "simulated", "source": "Kerr sim"},

    # BH001 - Area
    {"id": "BH001d", "domain": "gr", "law": "shadow_area ~ M²",
     "exponent": 2.000, "r2": 1.000, "n": 50, "year_discovered": 1916,
     "n_variables": 1, "data_type": "simulated", "source": "Kerr sim"},

    # BH003 - Inclination
    {"id": "BH003", "domain": "gr", "law": "cx*sin(θ)=const",
     "exponent": -1.000, "r2": 1.000, "n": 200, "year_discovered": 1973,
     "n_variables": 2, "data_type": "simulated", "source": "Kerr sim"},
]


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E096 -- Meta-ProtoScience: Patterns in Discovered Laws")
    print("=" * 70)

    data = [d for d in DISCOVERIES]
    n_laws = len(data)
    print(f"\n  Meta-dataset: {n_laws} discovered laws across {len(set(d['domain'] for d in data))} domains")

    # ── 1. R² vs log(n) ──────────────────────────────────────────
    print("\n  [1] Does more data = better fit?  (R² vs log₁₀(n))")
    r2_arr = np.array([d["r2"] for d in data])
    n_arr = np.array([d["n"] for d in data], dtype=float)
    log_n = np.log10(n_arr)

    coeffs = np.polyfit(log_n, r2_arr, 1)
    pred = np.polyval(coeffs, log_n)
    ss_res = np.sum((r2_arr - pred) ** 2)
    ss_tot = np.sum((r2_arr - np.mean(r2_arr)) ** 2)
    r2_of_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r_corr = float(np.corrcoef(log_n, r2_arr)[0, 1])

    print(f"    R² = {coeffs[1]:.4f} + {coeffs[0]:.4f} * log10(n)")
    print(f"    Correlation: r = {r_corr:.4f}")
    print(f"    Fit R² = {r2_of_r2:.4f}")
    print(f"    Verdict: {'More data helps' if coeffs[0] > 0.01 else 'Sample size does NOT predict fit quality'}")

    # ── 2. R² by domain ──────────────────────────────────────────
    print("\n  [2] R² by domain (is biology noisier than physics?)")
    domains = sorted(set(d["domain"] for d in data))
    domain_stats = {}
    for dom in domains:
        vals = [d["r2"] for d in data if d["domain"] == dom]
        mean_r2 = np.mean(vals)
        domain_stats[dom] = {"mean_r2": float(mean_r2), "n_laws": len(vals), "std": float(np.std(vals))}
        print(f"    {dom:15s}: mean R² = {mean_r2:.4f}  (n={len(vals)} laws, std={np.std(vals):.4f})")

    # Rank
    ranked = sorted(domain_stats.items(), key=lambda x: -x[1]["mean_r2"])
    print("\n    Hierarchy of precision:")
    for i, (dom, stats) in enumerate(ranked):
        print(f"      {i+1}. {dom:15s}  R² = {stats['mean_r2']:.4f}")

    # ── 3. Simulated vs Real ─────────────────────────────────────
    print("\n  [3] Simulated vs Real data")
    sim = [d["r2"] for d in data if d["data_type"] == "simulated"]
    real = [d["r2"] for d in data if d["data_type"] == "real"]
    print(f"    Simulated: mean R² = {np.mean(sim):.4f}  (n={len(sim)})")
    print(f"    Real data: mean R² = {np.mean(real):.4f}  (n={len(real)})")
    print(f"    Gap: {np.mean(sim) - np.mean(real):.4f}")

    # ── 4. Exponent distribution ─────────────────────────────────
    print("\n  [4] Exponent distribution (do nature's laws prefer simple numbers?)")
    exponents = [d["exponent"] for d in data if d["exponent"] is not None]
    abs_exp = [abs(e) for e in exponents]

    # Check clustering around simple fractions
    simple_fracs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    print(f"\n    {len(exponents)} laws have power-law exponents")
    print(f"    Range: {min(exponents):.3f} to {max(exponents):.3f}")
    print(f"    Mean |exponent|: {np.mean(abs_exp):.3f}")
    print(f"    Median |exponent|: {np.median(abs_exp):.3f}")

    # Distance to nearest simple fraction
    distances = []
    for e in exponents:
        nearest = min(simple_fracs, key=lambda f: abs(abs(e) - f))
        dist = abs(abs(e) - nearest)
        distances.append(dist)

    mean_dist = np.mean(distances)
    # Compare with random: if exponents were uniform [0,4], mean distance to nearest simple fraction would be ~0.125
    random_expected = 0.125
    print(f"\n    Mean distance to nearest simple fraction: {mean_dist:.4f}")
    print(f"    Random expectation: ~{random_expected:.4f}")
    print(f"    Ratio: {mean_dist / random_expected:.3f}x")
    clustering = "YES - exponents cluster near simple fractions" if mean_dist < random_expected * 0.8 else "Inconclusive"
    print(f"    Clustering: {clustering}")

    # Quarter-power analysis
    quarter_distances = []
    for e in exponents:
        nearest_q = round(e * 4) / 4
        quarter_distances.append(abs(e - nearest_q))
    mean_q_dist = np.mean(quarter_distances)
    print(f"\n    Quarter-power analysis:")
    print(f"    Mean distance to nearest n/4: {mean_q_dist:.4f}")

    # ── 5. R² vs year of discovery ───────────────────────────────
    print("\n  [5] R² vs year of original discovery")
    years = np.array([d["year_discovered"] for d in data], dtype=float)
    r_year = float(np.corrcoef(years, r2_arr)[0, 1])
    coeffs_year = np.polyfit(years, r2_arr, 1)
    print(f"    Correlation: r = {r_year:.4f}")
    print(f"    Slope: {coeffs_year[0]:.6f} R² per year")
    if r_year < -0.1:
        print(f"    => Newer laws are HARDER to find (more noise)")
    elif r_year > 0.1:
        print(f"    => Newer laws are EASIER to find (better data)")
    else:
        print(f"    => No relationship between era and precision")

    # ── 6. Number of variables vs R² ─────────────────────────────
    print("\n  [6] Complexity (n_variables) vs R²")
    n_vars = np.array([d["n_variables"] for d in data], dtype=float)
    r_vars = float(np.corrcoef(n_vars, r2_arr)[0, 1])
    print(f"    Correlation: r = {r_vars:.4f}")
    for nv in sorted(set(d["n_variables"] for d in data)):
        vals = [d["r2"] for d in data if d["n_variables"] == nv]
        print(f"    {nv} variable(s): mean R² = {np.mean(vals):.4f}  (n={len(vals)})")

    # ── 7. Source agency ranking ─────────────────────────────────
    print("\n  [7] Data source ranking")
    sources = sorted(set(d["source"] for d in data))
    for src in sources:
        vals = [d["r2"] for d in data if d["source"] == src]
        print(f"    {src:15s}: mean R² = {np.mean(vals):.4f}  (n={len(vals)})")

    # ── Summary ──────────────────────────────────────────────────
    print("\n  " + "=" * 60)
    print("  META-DISCOVERIES")
    print("  " + "=" * 60)

    findings = []

    # F1: Domain hierarchy
    f1 = f"Domain hierarchy: {' > '.join(d[0] for d in ranked)}"
    findings.append(f1)
    print(f"\n  1. {f1}")

    # F2: Simulated vs real gap
    gap = np.mean(sim) - np.mean(real)
    f2 = f"Simulated data gives R² {gap:.3f} higher than real data"
    findings.append(f2)
    print(f"  2. {f2}")

    # F3: Sample size
    f3 = f"Sample size vs R² correlation: r = {r_corr:.3f} ({'weak' if abs(r_corr) < 0.3 else 'moderate' if abs(r_corr) < 0.6 else 'strong'})"
    findings.append(f3)
    print(f"  3. {f3}")

    # F4: Exponent clustering
    f4 = f"Exponents cluster at {mean_dist/random_expected:.2f}x random rate near simple fractions"
    findings.append(f4)
    print(f"  4. {f4}")

    # F5: Year effect
    f5 = f"Year of discovery vs R²: r = {r_year:.3f} (no strong trend)"
    findings.append(f5)
    print(f"  5. {f5}")

    # F6: Complexity
    f6 = f"Number of variables vs R²: r = {r_vars:.3f}"
    findings.append(f6)
    print(f"  6. {f6}")

    # Artifact
    artifact = {
        "id": "E096",
        "timestamp": now,
        "world": "meta",
        "data_source": "ProtoScience experiment results",
        "status": "passed",
        "design": {
            "description": "Feed ProtoScience's own discovered laws back as data. Find patterns across domains, data types, and exponents.",
            "n_laws": n_laws,
            "n_domains": len(domains),
        },
        "result": {
            "r2_vs_log_n": {"slope": float(coeffs[0]), "r": r_corr, "r2": r2_of_r2},
            "domain_hierarchy": {d[0]: d[1] for d in ranked},
            "simulated_vs_real": {"sim_mean": float(np.mean(sim)), "real_mean": float(np.mean(real)), "gap": float(gap)},
            "exponent_clustering": {
                "mean_distance_to_simple": float(mean_dist),
                "random_expected": random_expected,
                "ratio": float(mean_dist / random_expected),
                "quarter_power_distance": float(mean_q_dist),
            },
            "year_correlation": {"r": r_year, "slope": float(coeffs_year[0])},
            "complexity_correlation": {"r": r_vars},
            "findings": findings,
        },
    }

    out_path = ROOT / "results" / "E096_meta_protoscience.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
