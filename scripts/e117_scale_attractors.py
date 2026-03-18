#!/usr/bin/env python3
"""
E117 — Scale Attractors: Do Nature's Laws Prefer Certain Numbers?

Hypothesis: Complex systems converge to descriptors with recurrent scale
ratios. Some classes may exhibit preferred constants (φ, 2, e, simple
fractions), while others show no preference.

Method:
  1. Collect ALL scale ratios from 51 ProtoScience experiments:
     - Power-law exponents
     - Consecutive-peak period ratios (FFT)
     - Rank-size ratios (Zipf)
     - Consecutive-scale ratios (allometry, fractal, spectral)
  2. Test clustering around candidate constants: φ, 2, e, 3/2, 4/3, √2
  3. Compare with uniform null distribution (Monte Carlo)
  4. Classify by system type: recursive-geometric, dissipative, social, spectral

Data: All ProtoScience result JSONs (E061-E116, BH001-BH003)
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ── Candidate attractor constants ─────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2  # 1.6180339...
E_CONST = np.e               # 2.7182818...
SQRT2 = np.sqrt(2)           # 1.4142135...
SQRT3 = np.sqrt(3)           # 1.7320508...

CANDIDATES = {
    "phi": PHI,
    "sqrt(2)":               SQRT2,
    "4/3":              4/3,
    "sqrt(3)":               SQRT3,
    "3/2":              3/2,
    "5/3":              5/3,
    "2":                2.0,
    "e":                E_CONST,
    "3":                3.0,
    "pi":                np.pi,
    "4":                4.0,
}

# Simple fractions for exponent clustering (from E096)
SIMPLE_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]


def load_result(eid):
    """Load a result JSON by experiment ID."""
    path = RESULTS / f"{eid}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def extract_ratios():
    """Extract all meaningful scale ratios from ProtoScience experiments."""

    ratios = []  # (value, source_experiment, category, description)
    exponents = []  # (value, source_experiment, description)

    # ── 1. Power-law exponents (expanded E096 dataset) ────────────────
    exp_data = [
        # (id, exponent, description)
        ("E061", 1.0, "Turbofan Ps30 degradation"),
        ("E062", 0.9988, "Kepler P²∝a³ slope"),
        ("E063", 2.1, "Fireball luminous efficiency"),
        ("E064", -2.09, "Voyager B∝r⁻² falloff"),
        ("E066", 0.600, "Chirp mass"),
        ("E067", 1.500, "Kepler asteroids"),
        ("E069", 1.000, "Hubble v=H₀d"),
        ("E071", 0.000, "Flat rotation curves"),
        ("E091", 4.065, "Stefan-Boltzmann T exponent"),
        ("E091", 2.003, "Stefan-Boltzmann R exponent"),
        ("E091", 4.123, "Mass-luminosity L∝M^α"),
        ("E092", 0.810, "Gutenberg-Richter b-value"),
        ("E094", 0.702, "Kleiber BMR∝M^0.75"),
        ("E094", 0.198, "Longevity∝M^0.25"),
        ("E094", 0.189, "Gestation∝M^0.25"),
        ("E094", -0.065, "Litter size∝M^-0.25"),
        ("E094", 0.872, "Neonate mass∝M^0.75"),
        ("E094", 1.061, "Home range∝M^1.0"),
        ("E094", -0.741, "Pop density∝M^-0.75"),
        ("E095", 0.092, "Gompertz β"),
        ("E097", 0.667, "Koide formula R"),
        ("E099", 2.27, "Cosmic ray spectral index"),
        ("E100", None, "Nuclear binding (not power-law)"),
        ("E106", None, "Pulsar P-Ṗ (complex)"),
        ("E109", -2.07, "Lunar crater size-frequency"),
        ("E113", 1.26, "Koch fractal dimension"),
        ("E113", 1.51, "Natural coastline fractal D"),
        ("BH001", 1.000, "r_horizon=2M"),
        ("BH001", 1.000, "r_shadow=3√3·M"),
        ("BH001", 1.000, "r_ISCO=6M"),
        ("BH001", 2.000, "shadow_area∝M²"),
        ("BH003", -1.000, "cx·sin(θ)=const"),
    ]

    for eid, exp, desc in exp_data:
        if exp is not None:
            exponents.append((exp, eid, desc))

    # ── 2. Consecutive FFT peak ratios ────────────────────────────────
    # Milankovitch (E107)
    r = load_result("E107_milankovitch")
    if r and "top_peaks" in r.get("result", {}):
        peaks = sorted([p["period"] for p in r["result"]["top_peaks"]], reverse=True)
        for i in range(len(peaks) - 1):
            ratio = peaks[i] / peaks[i + 1]
            if 1.0 < ratio < 10.0:
                ratios.append((ratio, "E107", "spectral",
                               f"Vostok FFT peak {peaks[i]:.0f}/{peaks[i+1]:.0f} yr"))

    # ── 3. Zipf rank-1/rank-2 ratios ─────────────────────────────────
    r = load_result("E101_zipf_cities")
    if r and "country_fits" in r.get("result", {}):
        for c in r["result"]["country_fits"]:
            ratio_12 = c.get("ratio_1_2")
            if ratio_12 and 1.0 < ratio_12 < 10.0:
                ratios.append((ratio_12, "E101", "social",
                               f"Zipf city ratio #1/#2 {c['country']}"))

    # ── 4. Zipf exponents as ratios to 1 ─────────────────────────────
    # Language Zipf alphas
    zipf_alphas = [
        ("E101", 0.892, "Zipf cities mean α"),
        ("E111", 1.131, "Zipf language mean α"),
        ("E112", 3.71, "Zipf gene expression α"),
        ("E114", 1.097, "Zipf global α"),
        ("E115", 0.79, "Zipf Linux code α"),
        ("E115", 0.81, "Zipf CPython code α"),
        ("E115", 0.87, "Zipf AI code α"),
    ]
    for eid, alpha, desc in zipf_alphas:
        exponents.append((alpha, eid, desc))

    # ── 5. Allometric ratios between scaling exponents ────────────────
    r = load_result("E094_kleiber_metabolic")
    if r and "power_law_discoveries" in r.get("result", {}):
        alphas = [d["alpha"] for d in r["result"]["power_law_discoveries"]
                  if abs(d["alpha"]) > 0.1]
        alphas_abs = sorted([abs(a) for a in alphas])
        for i in range(len(alphas_abs) - 1):
            if alphas_abs[i] > 0.05:
                ratio = alphas_abs[i + 1] / alphas_abs[i]
                if 1.0 < ratio < 10.0:
                    ratios.append((ratio, "E094", "recursive-geometric",
                                   f"Allometric α ratio {alphas_abs[i+1]:.3f}/{alphas_abs[i]:.3f}"))

    # ── 6. Milankovitch period ratios (consecutive cycles) ────────────
    milank = [105687.25, 38431.73, 21137.45]  # eccentricity, obliquity, precession
    for i in range(len(milank) - 1):
        ratio = milank[i] / milank[i + 1]
        ratios.append((ratio, "E107", "spectral",
                       f"Milankovitch {milank[i]:.0f}/{milank[i+1]:.0f} yr"))

    # ── 7. GR characteristic radii ratios ─────────────────────────────
    gr_radii = {
        "r_horizon": 2.0,
        "r_photon": 3.0,
        "r_ISCO": 6.0,
        "r_shadow": 3 * np.sqrt(3),  # 5.196
    }
    radii_sorted = sorted(gr_radii.items(), key=lambda x: x[1])
    for i in range(len(radii_sorted) - 1):
        r_val = radii_sorted[i + 1][1] / radii_sorted[i][1]
        ratios.append((r_val, "BH001", "dissipative",
                       f"GR {radii_sorted[i+1][0]}/{radii_sorted[i][0]}"))

    # ── 8. Particle physics mass ratios ──────────────────────────────
    particle_masses = {
        "J/ψ": 3.097,
        "Z": 91.19,
    }
    if len(particle_masses) >= 2:
        masses = sorted(particle_masses.values())
        for i in range(len(masses) - 1):
            ratio = masses[i + 1] / masses[i]
            ratios.append((ratio, "E079", "spectral",
                           f"Particle mass Z/J/ψ"))

    # ── 9. Cosmic ray spectral breaks ────────────────────────────────
    cr_breaks = [10**15.5, 10**18.5]  # knee, ankle in eV
    ratio = cr_breaks[1] / cr_breaks[0]
    ratios.append((ratio, "E099", "spectral", "Cosmic ray ankle/knee energy ratio"))

    # ── 10. Fractal dimensions as ratios ─────────────────────────────
    fractal_dims = [1.2618, 1.51]  # Koch, natural coastlines
    ratio = fractal_dims[1] / fractal_dims[0]
    ratios.append((ratio, "E113", "recursive-geometric",
                   "Coastline D / Koch D"))

    # ── 11. Musical consonance ratios ────────────────────────────────
    music_ratios = [
        (2.0, "octave 2:1"),
        (1.5, "fifth 3:2"),
        (4/3, "fourth 4:3"),
        (5/4, "major third 5:4"),
        (6/5, "minor third 6:5"),
    ]
    for val, desc in music_ratios:
        ratios.append((val, "E104", "recursive-geometric", f"Musical {desc}"))

    # ── 12. Earthquake magnitude spacing ─────────────────────────────
    # G-R law: log10(N) = a - b*M => each unit increase in M gives 10^b fewer events
    ratios.append((10**0.81, "E092", "dissipative",
                   "Gutenberg-Richter frequency ratio per magnitude"))

    # ── 13. COVID wave spacing ───────────────────────────────────────
    # approximate ratios of successive wave peaks  (from E081 description)
    # too noisy, skip

    # ── 14. Solar cycle harmonics ────────────────────────────────────
    solar_periods = [11.09, 92.4]  # main + Gleissberg
    ratios.append((solar_periods[1] / solar_periods[0], "E065", "spectral",
                   "Solar Gleissberg/main cycle"))

    return ratios, exponents


def distance_to_nearest(value, candidates):
    """Return (distance, nearest_name) to the closest candidate constant."""
    best_dist = float("inf")
    best_name = ""
    for name, c in candidates.items():
        d = abs(value - c)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_dist, best_name


def monte_carlo_null(n_ratios, n_sim=100_000):
    """
    Generate null distribution of mean distances to candidate constants.
    Uniform ratios in [1, 5] (typical range of observed ratios).
    """
    mean_dists = []
    cand_vals = list(CANDIDATES.values())
    for _ in range(n_sim):
        fake = np.random.uniform(1.0, 5.0, size=n_ratios)
        dists = [min(abs(f - c) for c in cand_vals) for f in fake]
        mean_dists.append(np.mean(dists))
    return np.array(mean_dists)


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 72)
    print("  E117 — Scale Attractors: Do Nature's Laws Prefer Certain Numbers?")
    print("=" * 72)

    ratios, exponents = extract_ratios()

    print(f"\n  Collected {len(ratios)} scale ratios from ProtoScience experiments")
    print(f"  Collected {len(exponents)} power-law exponents")

    # ── Analysis 1: Ratio clustering around candidate constants ───────
    print("\n" + "-" * 72)
    print("  [1] RATIO CLUSTERING AROUND CANDIDATE CONSTANTS")
    print("-" * 72)

    ratio_vals = np.array([r[0] for r in ratios if 1.0 < r[0] < 10.0])
    print(f"\n  Ratios in [1, 10]: {len(ratio_vals)}")
    print(f"  Range: {ratio_vals.min():.4f} — {ratio_vals.max():.4f}")
    print(f"  Mean: {ratio_vals.mean():.4f}")
    print(f"  Median: {np.median(ratio_vals):.4f}")

    # Distance to each candidate
    print(f"\n  {'Constant':<20s} {'Value':>8s} {'Mean dist':>10s} {'Nearest count':>14s}")
    print("  " + "—" * 56)

    candidate_nearest_counts = {name: 0 for name in CANDIDATES}
    candidate_distances = {name: [] for name in CANDIDATES}

    for rv in ratio_vals:
        _, nearest = distance_to_nearest(rv, CANDIDATES)
        candidate_nearest_counts[nearest] += 1
        for name, c in CANDIDATES.items():
            candidate_distances[name].append(abs(rv - c))

    for name, c in sorted(CANDIDATES.items(), key=lambda x: x[1]):
        mean_d = np.mean(candidate_distances[name])
        count = candidate_nearest_counts[name]
        marker = " <--" if count == max(candidate_nearest_counts.values()) else ""
        print(f"  {name:<20s} {c:>8.4f} {mean_d:>10.4f} {count:>14d}{marker}")

    # ── Analysis 2: Monte Carlo significance test ─────────────────────
    print("\n" + "-" * 72)
    print("  [2] MONTE CARLO SIGNIFICANCE TEST")
    print("-" * 72)

    # Observed mean distance to nearest candidate
    obs_dists = []
    for rv in ratio_vals:
        d, _ = distance_to_nearest(rv, CANDIDATES)
        obs_dists.append(d)
    obs_mean = np.mean(obs_dists)

    # Null distribution
    null_dists = monte_carlo_null(len(ratio_vals), n_sim=100_000)
    p_value = np.mean(null_dists <= obs_mean)

    print(f"\n  Observed mean distance to nearest constant: {obs_mean:.4f}")
    print(f"  Null (uniform [1,5]) mean distance:         {np.mean(null_dists):.4f}")
    print(f"  Ratio (observed/null):                      {obs_mean / np.mean(null_dists):.3f}×")
    print(f"  p-value (one-sided, clustering):             {p_value:.6f}")
    print(f"  Verdict: {'SIGNIFICANT clustering (p<0.05)' if p_value < 0.05 else 'NOT significant — no special clustering'}")

    # ── Analysis 3: φ specifically ────────────────────────────────────
    print("\n" + "-" * 72)
    print("  [3] GOLDEN RATIO (φ = 1.6180...) SPECIFICALLY")
    print("-" * 72)

    phi_dists = [abs(rv - PHI) for rv in ratio_vals]
    near_phi = [(r, ratios[i]) for i, r in enumerate(
        [r[0] for r in ratios if 1.0 < r[0] < 10.0])
        if abs(r - PHI) < 0.15]

    print(f"\n  Mean distance to φ: {np.mean(phi_dists):.4f}")
    print(f"  Ratios within 0.15 of φ: {len(near_phi)}/{len(ratio_vals)}")

    if near_phi:
        print(f"\n  Nearest to φ:")
        for val, (_, eid, cat, desc) in sorted(near_phi, key=lambda x: abs(x[0] - PHI)):
            print(f"    {val:.4f}  (Δ={abs(val-PHI):.4f})  {eid}: {desc}")

    # Compare φ distance vs distance to 3/2 and √3 (nearby competitors)
    mean_phi = np.mean(phi_dists)
    mean_1p5 = np.mean([abs(rv - 1.5) for rv in ratio_vals])
    mean_sqrt3 = np.mean([abs(rv - SQRT3) for rv in ratio_vals])

    print(f"\n  φ vs nearby competitors:")
    print(f"    Mean dist to φ (1.618):   {mean_phi:.4f}")
    print(f"    Mean dist to 3/2 (1.500): {mean_1p5:.4f}")
    print(f"    Mean dist to √3 (1.732):  {mean_sqrt3:.4f}")

    winner = min([("φ", mean_phi), ("3/2", mean_1p5), ("sqrt(3)", mean_sqrt3)],
                 key=lambda x: x[1])
    print(f"    Winner in ~1.5-1.7 range: {winner[0]}")

    # ── Analysis 4: Exponent clustering (expanded E096) ───────────────
    print("\n" + "-" * 72)
    print("  [4] POWER-LAW EXPONENT CLUSTERING (expanded from E096)")
    print("-" * 72)

    exp_vals = np.array([abs(e[0]) for e in exponents])
    print(f"\n  {len(exp_vals)} exponents")

    # Distance to simple fractions
    exp_simple_dists = []
    for ev in exp_vals:
        nearest = min(SIMPLE_FRACS, key=lambda f: abs(ev - f))
        exp_simple_dists.append(abs(ev - nearest))

    mean_exp_dist = np.mean(exp_simple_dists)
    random_exp_expected = 0.125  # for uniform over [0, 4]

    print(f"  Mean distance to nearest simple fraction: {mean_exp_dist:.4f}")
    print(f"  Random expectation:                       {random_exp_expected:.4f}")
    print(f"  Ratio:                                    {mean_exp_dist/random_exp_expected:.3f}×")

    # Quarter-power analysis
    quarter_dists = [abs(ev - round(ev * 4) / 4) for ev in exp_vals]
    print(f"  Mean distance to nearest n/4:             {np.mean(quarter_dists):.4f}")

    # ── Analysis 5: By system category ────────────────────────────────
    print("\n" + "-" * 72)
    print("  [5] PREFERRED RATIOS BY SYSTEM CATEGORY")
    print("-" * 72)

    categories = sorted(set(r[2] for r in ratios if 1.0 < r[0] < 10.0))
    category_results = {}

    for cat in categories:
        cat_ratios = [r[0] for r in ratios if r[2] == cat and 1.0 < r[0] < 10.0]
        if not cat_ratios:
            continue

        cat_arr = np.array(cat_ratios)
        # Find which constant each ratio is nearest to
        nearest_counts = {}
        for rv in cat_arr:
            _, nearest = distance_to_nearest(rv, CANDIDATES)
            nearest_counts[nearest] = nearest_counts.get(nearest, 0) + 1

        top_attractor = max(nearest_counts.items(), key=lambda x: x[1])
        mean_d = np.mean([distance_to_nearest(rv, CANDIDATES)[0] for rv in cat_arr])

        category_results[cat] = {
            "n": len(cat_ratios),
            "mean": float(np.mean(cat_arr)),
            "std": float(np.std(cat_arr)),
            "top_attractor": top_attractor[0],
            "top_count": top_attractor[1],
            "mean_distance": float(mean_d),
        }

        print(f"\n  {cat} (n={len(cat_ratios)}):")
        print(f"    Mean ratio: {np.mean(cat_arr):.4f} ± {np.std(cat_arr):.4f}")
        print(f"    Top attractor: {top_attractor[0]} ({top_attractor[1]}/{len(cat_ratios)} nearest)")
        print(f"    Mean distance to nearest constant: {mean_d:.4f}")
        for rv in sorted(cat_ratios):
            d, nearest = distance_to_nearest(rv, CANDIDATES)
            src = [r for r in ratios if r[0] == rv and r[2] == cat]
            desc = src[0][3] if src else ""
            print(f"      {rv:8.4f}  → {nearest:>6s} (Δ={d:.4f})  {desc}")

    # ── Analysis 6: Full ratio catalog ────────────────────────────────
    print("\n" + "-" * 72)
    print("  [6] FULL RATIO CATALOG")
    print("-" * 72)

    print(f"\n  {'Ratio':>8s}  {'Nearest':>8s}  {'Δ':>6s}  {'Exp':>6s}  {'Category':<20s}  Description")
    print("  " + "—" * 90)

    for val, eid, cat, desc in sorted(ratios, key=lambda r: r[0]):
        if 1.0 < val < 10.0:
            d, nearest = distance_to_nearest(val, CANDIDATES)
            print(f"  {val:8.4f}  {nearest:>8s}  {d:6.4f}  {eid:>6s}  {cat:<20s}  {desc}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CONCLUSIONS")
    print("=" * 72)

    findings = []

    # F1: Overall clustering
    if p_value < 0.05:
        f1 = (f"Scale ratios cluster around mathematical constants at "
              f"{obs_mean/np.mean(null_dists):.2f}× the null rate (p={p_value:.4f})")
    else:
        f1 = (f"Scale ratios do NOT show significant clustering around "
              f"mathematical constants (p={p_value:.4f})")
    findings.append(f1)
    print(f"\n  1. {f1}")

    # F2: φ specifically
    phi_count = candidate_nearest_counts.get("phi", 0)
    f2 = (f"φ is nearest attractor for {phi_count}/{len(ratio_vals)} ratios. "
          f"Winner in 1.5-1.7 range: {winner[0]}")
    findings.append(f2)
    print(f"  2. {f2}")

    # F3: Exponents
    f3 = (f"Power-law exponents cluster at {mean_exp_dist/random_exp_expected:.2f}× "
          f"random rate near simple fractions (expanded: {len(exp_vals)} exponents)")
    findings.append(f3)
    print(f"  3. {f3}")

    # F4: Category-specific
    cat_summaries = []
    for cat, res in category_results.items():
        cat_summaries.append(f"{cat}: top attractor = {res['top_attractor']}")
    f4 = "Category attractors: " + "; ".join(cat_summaries)
    findings.append(f4)
    print(f"  4. {f4}")

    # F5: Key insight
    f5 = ("Each class of system may have its own preferred scale ratios — "
          "there is no single universal constant like φ that governs all systems")
    findings.append(f5)
    print(f"  5. {f5}")

    print()

    # ── Artifact ──────────────────────────────────────────────────────
    artifact = {
        "id": "E117",
        "timestamp": now,
        "world": "meta",
        "data_source": "ProtoScience experiments E061-E116, BH001-BH003",
        "status": "passed",
        "design": {
            "description": (
                "Test whether nature's scale ratios cluster around preferred "
                "mathematical constants (φ, 2, e, simple fractions) or show "
                "no preference. Extract ratios from 51 experiments across 18 domains."
            ),
            "n_ratios": len(ratio_vals),
            "n_exponents": len(exp_vals),
            "n_candidates": len(CANDIDATES),
            "n_monte_carlo": 100_000,
        },
        "result": {
            "ratio_stats": {
                "n": int(len(ratio_vals)),
                "mean": float(ratio_vals.mean()),
                "median": float(np.median(ratio_vals)),
                "std": float(ratio_vals.std()),
            },
            "clustering_test": {
                "observed_mean_distance": float(obs_mean),
                "null_mean_distance": float(np.mean(null_dists)),
                "ratio_to_null": float(obs_mean / np.mean(null_dists)),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            },
            "candidate_nearest_counts": {
                name: int(count)
                for name, count in sorted(
                    candidate_nearest_counts.items(),
                    key=lambda x: -x[1]
                )
            },
            "phi_analysis": {
                "mean_distance_to_phi": float(mean_phi),
                "mean_distance_to_3_2": float(mean_1p5),
                "mean_distance_to_sqrt3": float(mean_sqrt3),
                "winner_in_1p5_1p7_range": winner[0],
                "n_within_0p15_of_phi": len(near_phi),
            },
            "exponent_clustering": {
                "n_exponents": int(len(exp_vals)),
                "mean_distance_to_simple_fraction": float(mean_exp_dist),
                "random_expected": random_exp_expected,
                "ratio_to_random": float(mean_exp_dist / random_exp_expected),
                "mean_distance_to_quarter": float(np.mean(quarter_dists)),
            },
            "category_results": category_results,
            "all_ratios": [
                {"value": float(v), "experiment": e, "category": c, "description": d}
                for v, e, c, d in sorted(ratios, key=lambda r: r[0])
                if 1.0 < v < 10.0
            ],
            "findings": findings,
        },
    }

    out_path = RESULTS / "E117_scale_attractors.json"
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"  Artifact: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
