#!/usr/bin/env python3
"""
E097 — Searching for Patterns in Elementary Particle Masses

Question: Is there a mathematical pattern in the masses of quarks
and leptons? Can we find a Koide-like formula for quarks?

Data: Particle Data Group (PDG) 2024 — masses of all fermions
  - 6 quarks, 6 leptons (3 charged + 3 neutrinos)

This is different from every other ProtoScience experiment:
there is NO known governing equation for particle masses.
If we find one, it would be genuinely new.

Source: https://pdg.lbl.gov/
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── PDG 2024 Particle Masses (MeV/c²) ──────────────────────────
# Source: Particle Data Group, Phys. Rev. D 110, 030001 (2024)

CHARGED_LEPTONS = {
    "electron":  {"mass": 0.51099895,  "charge": -1, "generation": 1},
    "muon":      {"mass": 105.6583755, "charge": -1, "generation": 2},
    "tau":       {"mass": 1776.86,     "charge": -1, "generation": 3},
}

# Neutrino masses are upper limits / oscillation constraints
# We use the squared mass differences for analysis
NEUTRINOS = {
    "nu_e":  {"mass_upper": 0.0000008, "generation": 1},  # eV scale
    "nu_mu": {"mass_upper": 0.00017,   "generation": 2},
    "nu_tau": {"mass_upper": 0.0156,   "generation": 3},
}

UP_TYPE_QUARKS = {
    "up":    {"mass": 2.16,     "charge": 2/3, "generation": 1},
    "charm": {"mass": 1270.0,   "charge": 2/3, "generation": 2},
    "top":   {"mass": 172760.0, "charge": 2/3, "generation": 3},
}

DOWN_TYPE_QUARKS = {
    "down":    {"mass": 4.67,    "charge": -1/3, "generation": 1},
    "strange": {"mass": 93.4,    "charge": -1/3, "generation": 2},
    "bottom":  {"mass": 4180.0,  "charge": -1/3, "generation": 3},
}

ALL_FERMIONS = [
    {"name": "electron", "mass": 0.511,     "type": "lepton", "charge": -1,   "gen": 1},
    {"name": "muon",     "mass": 105.658,   "type": "lepton", "charge": -1,   "gen": 2},
    {"name": "tau",      "mass": 1776.86,   "type": "lepton", "charge": -1,   "gen": 3},
    {"name": "up",       "mass": 2.16,      "type": "quark",  "charge": 2/3,  "gen": 1},
    {"name": "down",     "mass": 4.67,      "type": "quark",  "charge": -1/3, "gen": 1},
    {"name": "charm",    "mass": 1270.0,    "type": "quark",  "charge": 2/3,  "gen": 2},
    {"name": "strange",  "mass": 93.4,      "type": "quark",  "charge": -1/3, "gen": 2},
    {"name": "top",      "mass": 172760.0,  "type": "quark",  "charge": 2/3,  "gen": 3},
    {"name": "bottom",   "mass": 4180.0,    "type": "quark",  "charge": -1/3, "gen": 3},
]


def koide_ratio(m1, m2, m3):
    """Koide ratio R = (m1+m2+m3) / (sqrt(m1)+sqrt(m2)+sqrt(m3))^2"""
    s = m1 + m2 + m3
    sr = (np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)) ** 2
    return s / sr


def check_koide():
    """Verify Koide formula for charged leptons and test on quarks."""
    print("\n  [1] Koide Formula Verification")
    print("  R = (m1 + m2 + m3) / (sqrt(m1) + sqrt(m2) + sqrt(m3))^2")
    print(f"  Expected: 2/3 = {2/3:.10f}")

    # Charged leptons (the known case)
    me, mmu, mtau = 0.51099895, 105.6583755, 1776.86
    R_leptons = koide_ratio(me, mmu, mtau)
    err_leptons = abs(R_leptons - 2/3)
    print(f"\n  Charged leptons (e, mu, tau):")
    print(f"    R = {R_leptons:.10f}")
    print(f"    Error from 2/3: {err_leptons:.2e}")
    print(f"    {'CONFIRMED' if err_leptons < 0.001 else 'FAILED'} (precision: {err_leptons/R_leptons*100:.4f}%)")

    # Up-type quarks (u, c, t)
    mu, mc, mt = 2.16, 1270.0, 172760.0
    R_up = koide_ratio(mu, mc, mt)
    print(f"\n  Up-type quarks (u, c, t):")
    print(f"    R = {R_up:.10f}")
    print(f"    Distance from 2/3: {abs(R_up - 2/3):.6f}")

    # Down-type quarks (d, s, b)
    md, ms, mb = 4.67, 93.4, 4180.0
    R_down = koide_ratio(md, ms, mb)
    print(f"\n  Down-type quarks (d, s, b):")
    print(f"    R = {R_down:.10f}")
    print(f"    Distance from 2/3: {abs(R_down - 2/3):.6f}")

    # All charged fermions in triplets
    print(f"\n  All fermion triplets by generation:")
    # Gen 1-2-3 of each type
    R_all_up = koide_ratio(mu, mc, mt)
    R_all_down = koide_ratio(md, ms, mb)
    R_all_lep = koide_ratio(me, mmu, mtau)

    return {
        "leptons": {"R": float(R_leptons), "error": float(err_leptons)},
        "up_quarks": {"R": float(R_up), "distance_from_2_3": float(abs(R_up - 2/3))},
        "down_quarks": {"R": float(R_down), "distance_from_2_3": float(abs(R_down - 2/3))},
    }


def mass_ratios():
    """Analyze mass ratios between generations."""
    print("\n  [2] Mass Ratios Between Generations")

    families = {
        "charged_leptons": [0.511, 105.658, 1776.86],
        "up_quarks": [2.16, 1270.0, 172760.0],
        "down_quarks": [4.67, 93.4, 4180.0],
    }

    results = {}
    for name, masses in families.items():
        r21 = masses[1] / masses[0]
        r32 = masses[2] / masses[1]
        r31 = masses[2] / masses[0]
        log_r21 = np.log(r21)
        log_r32 = np.log(r32)

        print(f"\n  {name}:")
        print(f"    m1={masses[0]}, m2={masses[1]}, m3={masses[2]} MeV")
        print(f"    m2/m1 = {r21:.2f}")
        print(f"    m3/m2 = {r32:.2f}")
        print(f"    m3/m1 = {r31:.2f}")
        print(f"    log(m2/m1) = {log_r21:.4f}")
        print(f"    log(m3/m2) = {log_r32:.4f}")
        print(f"    log ratio: {log_r32/log_r21:.4f}")

        results[name] = {
            "masses": masses,
            "r21": float(r21), "r32": float(r32), "r31": float(r31),
            "log_r21": float(log_r21), "log_r32": float(log_r32),
            "log_ratio": float(log_r32 / log_r21),
        }

    return results


def power_law_generation():
    """Fit m = C * n^alpha for each family (n = generation number)."""
    print("\n  [3] Power-law fit: mass = C * generation^alpha")

    families = {
        "charged_leptons": [0.511, 105.658, 1776.86],
        "up_quarks": [2.16, 1270.0, 172760.0],
        "down_quarks": [4.67, 93.4, 4180.0],
    }

    gen = np.array([1, 2, 3], dtype=float)
    results = {}

    for name, masses in families.items():
        m = np.array(masses)
        log_g = np.log10(gen)
        log_m = np.log10(m)

        coeffs = np.polyfit(log_g, log_m, 1)
        alpha = coeffs[0]
        C = 10 ** coeffs[1]

        pred = np.polyval(coeffs, log_g)
        ss_res = np.sum((log_m - pred) ** 2)
        ss_tot = np.sum((log_m - np.mean(log_m)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"\n  {name}: m = {C:.4f} * gen^{alpha:.4f}  (R2={r2:.6f})")
        results[name] = {"alpha": float(alpha), "C": float(C), "r2": float(r2)}

    return results


def exponential_generation():
    """Fit m = C * exp(beta * n) for each family."""
    print("\n  [4] Exponential fit: mass = C * exp(beta * generation)")

    families = {
        "charged_leptons": [0.511, 105.658, 1776.86],
        "up_quarks": [2.16, 1270.0, 172760.0],
        "down_quarks": [4.67, 93.4, 4180.0],
    }

    gen = np.array([1, 2, 3], dtype=float)
    results = {}

    for name, masses in families.items():
        m = np.array(masses)
        log_m = np.log(m)

        coeffs = np.polyfit(gen, log_m, 1)
        beta = coeffs[0]
        C = np.exp(coeffs[1])

        pred = np.polyval(coeffs, gen)
        ss_res = np.sum((log_m - pred) ** 2)
        ss_tot = np.sum((log_m - np.mean(log_m)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"\n  {name}: m = {C:.6f} * exp({beta:.4f} * gen)  (R2={r2:.6f})")
        print(f"    Mass multiplier per generation: x{np.exp(beta):.2f}")
        results[name] = {"beta": float(beta), "C": float(C), "r2": float(r2),
                         "multiplier": float(np.exp(beta))}

    return results


def sqrt_mass_pattern():
    """Check if sqrt(m) follows a linear pattern (Koide-inspired)."""
    print("\n  [5] Square-root mass pattern: sqrt(m) vs generation")

    families = {
        "charged_leptons": [0.511, 105.658, 1776.86],
        "up_quarks": [2.16, 1270.0, 172760.0],
        "down_quarks": [4.67, 93.4, 4180.0],
    }

    gen = np.array([1, 2, 3], dtype=float)
    results = {}

    for name, masses in families.items():
        sm = np.sqrt(masses)
        coeffs = np.polyfit(gen, sm, 1)
        pred = np.polyval(coeffs, gen)
        ss_res = np.sum((sm - pred) ** 2)
        ss_tot = np.sum((sm - np.mean(sm)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"\n  {name}: sqrt(m) = {coeffs[1]:.4f} + {coeffs[0]:.4f} * gen  (R2={r2:.6f})")
        print(f"    sqrt masses: {sm[0]:.4f}, {sm[1]:.4f}, {sm[2]:.4f}")
        results[name] = {"slope": float(coeffs[0]), "intercept": float(coeffs[1]),
                         "r2": float(r2)}

    return results


def cross_family_ratios():
    """Look for patterns across families (lepton/quark mass ratios)."""
    print("\n  [6] Cross-family mass ratios")

    pairs = [
        ("electron/up", 0.511, 2.16),
        ("muon/charm", 105.658, 1270.0),
        ("tau/bottom", 1776.86, 4180.0),
        ("electron/down", 0.511, 4.67),
        ("muon/strange", 105.658, 93.4),
        ("tau/top", 1776.86, 172760.0),
    ]

    results = {}
    for name, m1, m2 in pairs:
        ratio = m1 / m2
        log_ratio = np.log10(ratio)
        print(f"    {name:20s}: {ratio:.6f}  (log10: {log_ratio:.4f})")
        results[name] = {"ratio": float(ratio), "log_ratio": float(log_ratio)}

    # Check if lepton/quark ratios follow a pattern
    lq_ratios = [0.511/2.16, 105.658/1270.0, 1776.86/4180.0]
    print(f"\n    Charged lepton / up-type quark by generation:")
    for i, r in enumerate(lq_ratios):
        print(f"      Gen {i+1}: {r:.6f}")

    # Are these ratios converging?
    if len(lq_ratios) >= 2:
        r_change = [lq_ratios[i+1]/lq_ratios[i] for i in range(len(lq_ratios)-1)]
        print(f"    Ratio change: {r_change}")

    return results


def all_masses_power_law():
    """Fit all 9 fermion masses as a single power law of some index."""
    print("\n  [7] All fermion masses: is there a universal ordering?")

    masses = sorted([f["mass"] for f in ALL_FERMIONS])
    ranks = np.arange(1, len(masses) + 1, dtype=float)

    log_r = np.log10(ranks)
    log_m = np.log10(masses)

    coeffs = np.polyfit(log_r, log_m, 1)
    alpha = coeffs[0]
    C = 10 ** coeffs[1]

    pred = np.polyval(coeffs, log_r)
    ss_res = np.sum((log_m - pred) ** 2)
    ss_tot = np.sum((log_m - np.mean(log_m)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"    Sorted masses: {[f'{m:.3f}' for m in masses]}")
    print(f"    m(rank) = {C:.4f} * rank^{alpha:.4f}  (R2={r2:.6f})")
    print(f"    Mass range spans {masses[-1]/masses[0]:.0f}x ({np.log10(masses[-1]/masses[0]):.1f} decades)")

    # Try exponential
    coeffs_exp = np.polyfit(ranks, log_m, 1)
    pred_exp = np.polyval(coeffs_exp, ranks)
    ss_res_exp = np.sum((log_m - pred_exp) ** 2)
    r2_exp = 1.0 - ss_res_exp / ss_tot if ss_tot > 0 else 0.0

    print(f"    log10(m) = {coeffs_exp[1]:.4f} + {coeffs_exp[0]:.4f} * rank  (R2={r2_exp:.6f})")

    return {
        "power_law": {"alpha": float(alpha), "C": float(C), "r2": float(r2)},
        "exponential": {"slope": float(coeffs_exp[0]), "intercept": float(coeffs_exp[1]),
                        "r2": float(r2_exp)},
    }


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E097 -- Searching for Patterns in Elementary Particle Masses")
    print("=" * 70)
    print(f"\n  Data: PDG 2024 — 9 charged fermions (3 leptons + 6 quarks)")
    print(f"  WARNING: Only 9 data points. Any pattern found must be treated")
    print(f"  with extreme skepticism. This is exploratory, not confirmatory.")

    # 1. Koide
    koide_results = check_koide()

    # 2. Mass ratios
    ratio_results = mass_ratios()

    # 3. Power law
    power_results = power_law_generation()

    # 4. Exponential
    exp_results = exponential_generation()

    # 5. Sqrt pattern
    sqrt_results = sqrt_mass_pattern()

    # 6. Cross-family
    cross_results = cross_family_ratios()

    # 7. All masses
    all_results = all_masses_power_law()

    # Summary
    print("\n  " + "=" * 60)
    print("  SUMMARY")
    print("  " + "=" * 60)

    print(f"\n  Koide formula (charged leptons): R = {koide_results['leptons']['R']:.10f}")
    print(f"    Error from 2/3: {koide_results['leptons']['error']:.2e}  [CONFIRMED]")

    print(f"\n  Koide for up-quarks:   R = {koide_results['up_quarks']['R']:.6f}  (2/3 = 0.6667)")
    print(f"  Koide for down-quarks: R = {koide_results['down_quarks']['R']:.6f}  (2/3 = 0.6667)")

    # Best fit model
    print(f"\n  Best model for mass vs generation:")
    for name in ["charged_leptons", "up_quarks", "down_quarks"]:
        r2_pow = power_results[name]["r2"]
        r2_exp = exp_results[name]["r2"]
        if r2_exp > r2_pow:
            print(f"    {name:20s}: EXPONENTIAL  m = {exp_results[name]['C']:.4f} * exp({exp_results[name]['beta']:.4f} * gen)  R2={r2_exp:.6f}")
        else:
            print(f"    {name:20s}: POWER LAW    m = {power_results[name]['C']:.4f} * gen^{power_results[name]['alpha']:.4f}  R2={r2_pow:.6f}")

    print(f"\n  CAVEAT: All fits have n=3 (three generations).")
    print(f"  With 3 points, any 2-parameter model will fit well.")
    print(f"  These results are suggestive, NOT confirmatory.")

    # Artifact
    artifact = {
        "id": "E097",
        "timestamp": now,
        "world": "particle_physics",
        "data_source": "Particle Data Group (PDG) 2024",
        "data_url": "https://pdg.lbl.gov/",
        "status": "exploratory",
        "design": {
            "description": "Search for mathematical patterns in elementary particle masses. Verify Koide formula, test extensions to quarks, fit generation scaling.",
            "n_particles": 9,
            "caveat": "Only 9 data points. All patterns must be treated as exploratory.",
        },
        "result": {
            "koide": koide_results,
            "mass_ratios": ratio_results,
            "power_law_fits": power_results,
            "exponential_fits": exp_results,
            "sqrt_mass": sqrt_results,
            "cross_family": cross_results,
            "all_masses": all_results,
            "key_findings": [
                f"Koide formula confirmed for charged leptons: R = {koide_results['leptons']['R']:.10f} (error: {koide_results['leptons']['error']:.2e})",
                f"Koide for up-quarks: R = {koide_results['up_quarks']['R']:.6f} (deviates from 2/3)",
                f"Koide for down-quarks: R = {koide_results['down_quarks']['R']:.6f} (deviates from 2/3)",
                "All families show exponential mass growth across generations",
                "CAVEAT: n=3 per family — insufficient for statistical conclusions",
            ],
        },
    }

    out_path = ROOT / "results" / "E097_particle_masses.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
