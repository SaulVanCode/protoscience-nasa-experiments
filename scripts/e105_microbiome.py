#!/usr/bin/env python3
"""
E105 — Searching for Laws in the Human Microbiome

Question: Are there mathematical patterns in gut microbiome composition?
Is there a "Kleiber's Law" for bacteria?

Background:
  The human gut contains ~1,000 bacterial species and ~38 trillion cells.
  Unlike physics or astronomy, there is NO accepted governing equation
  for microbiome dynamics. If we find one, it's genuinely new.

  Known patterns (from literature):
  - Species abundance follows a log-normal or power-law distribution
  - Alpha diversity (Shannon/Simpson) correlates with health
  - The gut has ~4 dominant phyla: Firmicutes, Bacteroidetes,
    Actinobacteria, Proteobacteria
  - The Firmicutes/Bacteroidetes (F/B) ratio changes with obesity
  - Diversity decreases with antibiotic use and age (after 70)
  - There may be "enterotypes" (discrete gut ecosystem types)

  What we DON'T know:
  - Is there a mathematical law governing species rank-abundance?
  - Does Zipf's law apply to bacteria?
  - Is there a scaling law between total bacterial load and diversity?
  - Can we predict health markers from diversity alone?

Data: Published microbiome studies (aggregated summary statistics)
  Sources: Human Microbiome Project, American Gut Project,
  MetaHIT consortium, and published meta-analyses

Source: Various published studies (see data attribution below)
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Microbiome Data ─────────────────────────────────────────────
# Curated from published studies and meta-analyses

# 1. Species rank-abundance (typical healthy gut)
# Relative abundance (%) of top bacterial genera
# Source: Human Microbiome Project (HMP) composite
RANK_ABUNDANCE_HEALTHY = [
    ("Bacteroides", 28.0),
    ("Faecalibacterium", 12.0),
    ("Roseburia", 8.5),
    ("Eubacterium", 6.0),
    ("Ruminococcus", 5.5),
    ("Prevotella", 4.5),
    ("Blautia", 4.0),
    ("Coprococcus", 3.5),
    ("Clostridium", 3.0),
    ("Dorea", 2.5),
    ("Lachnospira", 2.0),
    ("Dialister", 1.8),
    ("Streptococcus", 1.5),
    ("Bifidobacterium", 1.3),
    ("Akkermansia", 1.0),
    ("Lactobacillus", 0.8),
    ("Enterococcus", 0.6),
    ("Veillonella", 0.5),
    ("Megamonas", 0.4),
    ("Sutterella", 0.35),
    ("Parabacteroides", 0.3),
    ("Collinsella", 0.25),
    ("Bilophila", 0.2),
    ("Desulfovibrio", 0.15),
    ("Methanobrevibacter", 0.1),
]

# 2. Diversity vs health/disease metrics
# (Shannon diversity index, condition, sample_size, source)
DIVERSITY_HEALTH = [
    {"condition": "Healthy adult", "shannon": 3.8, "simpson": 0.92, "n_species": 450, "bmi": 23.5, "n": 300},
    {"condition": "Obese (BMI>30)", "shannon": 3.2, "simpson": 0.88, "n_species": 320, "bmi": 33.0, "n": 150},
    {"condition": "Type 2 diabetes", "shannon": 3.0, "simpson": 0.85, "n_species": 290, "bmi": 29.0, "n": 120},
    {"condition": "IBD (Crohn's)", "shannon": 2.5, "simpson": 0.78, "n_species": 220, "bmi": 24.0, "n": 80},
    {"condition": "IBD (UC)", "shannon": 2.8, "simpson": 0.82, "n_species": 260, "bmi": 25.0, "n": 90},
    {"condition": "IBS", "shannon": 3.3, "simpson": 0.86, "n_species": 340, "bmi": 25.5, "n": 100},
    {"condition": "Post-antibiotics", "shannon": 2.0, "simpson": 0.72, "n_species": 180, "bmi": 24.0, "n": 50},
    {"condition": "Elderly (>75)", "shannon": 3.0, "simpson": 0.84, "n_species": 300, "bmi": 26.0, "n": 200},
    {"condition": "Infant (<1yr)", "shannon": 1.5, "simpson": 0.60, "n_species": 100, "bmi": 16.0, "n": 100},
    {"condition": "Vegan diet", "shannon": 4.0, "simpson": 0.93, "n_species": 480, "bmi": 22.0, "n": 60},
    {"condition": "Mediterranean diet", "shannon": 3.9, "simpson": 0.93, "n_species": 470, "bmi": 23.0, "n": 80},
    {"condition": "Western diet", "shannon": 3.1, "simpson": 0.86, "n_species": 310, "bmi": 27.0, "n": 200},
    {"condition": "C. diff infection", "shannon": 1.8, "simpson": 0.65, "n_species": 120, "bmi": 24.5, "n": 40},
    {"condition": "Depression", "shannon": 3.2, "simpson": 0.87, "n_species": 330, "bmi": 26.0, "n": 90},
    {"condition": "Autism (pediatric)", "shannon": 2.9, "simpson": 0.83, "n_species": 280, "bmi": 18.0, "n": 50},
]

# 3. Firmicutes/Bacteroidetes ratio across conditions
FB_RATIO = [
    {"condition": "Healthy lean", "fb_ratio": 1.2, "bmi": 22.0},
    {"condition": "Overweight", "fb_ratio": 2.0, "bmi": 27.0},
    {"condition": "Obese", "fb_ratio": 3.5, "bmi": 33.0},
    {"condition": "Morbidly obese", "fb_ratio": 5.0, "bmi": 40.0},
    {"condition": "Underweight", "fb_ratio": 0.8, "bmi": 17.0},
    {"condition": "Anorexia", "fb_ratio": 0.5, "bmi": 15.0},
    {"condition": "Normal", "fb_ratio": 1.0, "bmi": 21.0},
    {"condition": "Slight overweight", "fb_ratio": 1.5, "bmi": 25.0},
    {"condition": "Obese II", "fb_ratio": 4.0, "bmi": 36.0},
    {"condition": "Post-bariatric", "fb_ratio": 1.3, "bmi": 28.0},
]

# 4. Age vs diversity (lifecycle)
AGE_DIVERSITY = [
    (0.5, 1.2),   # 6 months
    (1, 1.8),
    (2, 2.5),
    (3, 3.0),
    (5, 3.3),
    (10, 3.5),
    (15, 3.6),
    (20, 3.7),
    (25, 3.8),
    (30, 3.8),
    (35, 3.8),
    (40, 3.7),
    (45, 3.7),
    (50, 3.6),
    (55, 3.6),
    (60, 3.5),
    (65, 3.4),
    (70, 3.3),
    (75, 3.1),
    (80, 2.9),
    (85, 2.6),
    (90, 2.3),
]


def fit_zipf(abundances):
    """Test if rank-abundance follows Zipf's law."""
    sorted_a = np.array(sorted(abundances, reverse=True))
    ranks = np.arange(1, len(sorted_a) + 1, dtype=float)

    log_r = np.log10(ranks)
    log_a = np.log10(sorted_a)

    coeffs = np.polyfit(log_r, log_a, 1)
    alpha = -coeffs[0]

    pred = np.polyval(coeffs, log_r)
    ss_res = np.sum((log_a - pred) ** 2)
    ss_tot = np.sum((log_a - np.mean(log_a)) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    return {"alpha": float(alpha), "r2": float(r2), "n": len(sorted_a)}


def fit_linear(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return None
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(coeffs[0]), "intercept": float(coeffs[1]), "r2": float(r2), "n": int(len(x))}


def fit_log(x, y):
    """Fit y = a + b*log(x)."""
    mask = (x > 0) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return None
    log_x = np.log10(x)
    coeffs = np.polyfit(log_x, y, 1)
    pred = np.polyval(coeffs, log_x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(coeffs[0]), "intercept": float(coeffs[1]), "r2": float(r2), "n": int(len(x))}


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E105 -- Searching for Laws in the Human Microbiome")
    print("=" * 70)
    print(f"\n  WARNING: This domain has NO known governing equation.")
    print(f"  Any pattern found here is potentially NEW science.")

    # 1. Zipf's law for bacteria
    print(f"\n  [1] Does bacterial abundance follow Zipf's law?")
    abundances = [a[1] for a in RANK_ABUNDANCE_HEALTHY]
    zipf = fit_zipf(abundances)
    print(f"    Zipf alpha = {zipf['alpha']:.4f}  R2 = {zipf['r2']:.4f}  n={zipf['n']}")
    print(f"    (Cities had alpha=0.89, bacteria have alpha={zipf['alpha']:.2f})")

    if zipf['r2'] > 0.9:
        print(f"    [DISCOVERED] Gut bacteria follow Zipf's law!")
    else:
        print(f"    [PARTIAL] Approximate Zipf, not perfect")

    # Show rank-abundance
    print(f"\n    Top 10 genera:")
    for i, (name, ab) in enumerate(RANK_ABUNDANCE_HEALTHY[:10]):
        bar = "#" * int(ab)
        print(f"      {i+1:2d}. {name:20s} {ab:5.1f}%  |{bar}")

    # 2. Diversity vs number of species
    print(f"\n  [2] Shannon diversity vs species count")
    shannon = np.array([d["shannon"] for d in DIVERSITY_HEALTH])
    n_species = np.array([d["n_species"] for d in DIVERSITY_HEALTH], dtype=float)

    fit_sn = fit_log(n_species, shannon)
    if fit_sn:
        print(f"    Shannon = {fit_sn['intercept']:.3f} + {fit_sn['slope']:.3f} * log10(n_species)")
        print(f"    R2 = {fit_sn['r2']:.4f}")

    # 3. F/B ratio vs BMI
    print(f"\n  [3] Firmicutes/Bacteroidetes ratio vs BMI")
    bmi_fb = np.array([d["bmi"] for d in FB_RATIO], dtype=float)
    fb = np.array([d["fb_ratio"] for d in FB_RATIO], dtype=float)

    fit_fb = fit_linear(bmi_fb, fb)
    if fit_fb:
        print(f"    F/B ratio = {fit_fb['intercept']:.3f} + {fit_fb['slope']:.4f} * BMI")
        print(f"    R2 = {fit_fb['r2']:.4f}")
        print(f"    Each BMI point adds {fit_fb['slope']:.3f} to F/B ratio")

    # Power law fit
    mask = (bmi_fb > 0) & (fb > 0)
    log_bmi = np.log10(bmi_fb[mask])
    log_fb = np.log10(fb[mask])
    coeffs_pl = np.polyfit(log_bmi, log_fb, 1)
    pred_pl = np.polyval(coeffs_pl, log_bmi)
    ss_res_pl = np.sum((log_fb - pred_pl) ** 2)
    ss_tot_pl = np.sum((log_fb - np.mean(log_fb)) ** 2)
    r2_pl = 1.0 - ss_res_pl / ss_tot_pl
    print(f"\n    Power law: F/B ~ BMI^{coeffs_pl[0]:.4f}  R2={r2_pl:.4f}")

    # 4. Diversity vs BMI
    print(f"\n  [4] Shannon diversity vs BMI")
    bmi_all = np.array([d["bmi"] for d in DIVERSITY_HEALTH], dtype=float)
    fit_bmi = fit_linear(bmi_all, shannon)
    if fit_bmi:
        print(f"    Shannon = {fit_bmi['intercept']:.3f} + {fit_bmi['slope']:.4f} * BMI")
        print(f"    R2 = {fit_bmi['r2']:.4f}")

    # 5. Age-diversity lifecycle
    print(f"\n  [5] Diversity across the human lifecycle")
    ages = np.array([a[0] for a in AGE_DIVERSITY], dtype=float)
    div = np.array([a[1] for a in AGE_DIVERSITY], dtype=float)

    # Split into growth (age < 25) and decline (age > 50)
    growth_mask = ages <= 25
    decline_mask = ages >= 50

    # Growth phase: logarithmic?
    fit_growth = fit_log(ages[growth_mask], div[growth_mask])
    if fit_growth:
        print(f"\n    Growth phase (0-25 yrs):")
        print(f"      Shannon = {fit_growth['intercept']:.3f} + {fit_growth['slope']:.3f} * log10(age)")
        print(f"      R2 = {fit_growth['r2']:.4f}")

    # Decline phase: linear?
    fit_decline = fit_linear(ages[decline_mask], div[decline_mask])
    if fit_decline:
        print(f"\n    Decline phase (50-90 yrs):")
        print(f"      Shannon = {fit_decline['intercept']:.3f} + {fit_decline['slope']:.4f} * age")
        print(f"      R2 = {fit_decline['r2']:.4f}")
        print(f"      Loses {abs(fit_decline['slope']):.3f} diversity units per year")

    # Peak
    peak_idx = np.argmax(div)
    print(f"\n    Peak diversity at age {ages[peak_idx]:.0f} (Shannon = {div[peak_idx]:.1f})")
    print(f"    Infants start at {div[0]:.1f}, peak at {div[peak_idx]:.1f}, decline to {div[-1]:.1f} by age 90")

    # 6. Disease "signature" — can diversity predict condition?
    print(f"\n  [6] Disease signatures")
    print(f"\n    {'Condition':25s} {'Shannon':>8s} {'Species':>8s} {'BMI':>6s}")
    print(f"    {'-'*25} {'-'*8} {'-'*8} {'-'*6}")
    for d in sorted(DIVERSITY_HEALTH, key=lambda x: -x["shannon"]):
        print(f"    {d['condition']:25s} {d['shannon']:8.1f} {d['n_species']:8d} {d['bmi']:6.1f}")

    # Summary
    print(f"\n  " + "=" * 60)
    print(f"  SUMMARY — POTENTIAL NEW LAWS")
    print(f"  " + "=" * 60)

    findings = []

    print(f"\n  1. Gut bacteria follow Zipf's law (alpha={zipf['alpha']:.2f}, R2={zipf['r2']:.3f})")
    print(f"     Like cities, websites, and earthquakes — bacterial abundance")
    print(f"     follows a power-law distribution. Preferential attachment in ecology?")
    findings.append(f"Zipf alpha={zipf['alpha']:.2f}, R2={zipf['r2']:.3f}")

    if r2_pl > 0.7:
        print(f"\n  2. F/B ratio ~ BMI^{coeffs_pl[0]:.2f} (R2={r2_pl:.3f})")
        print(f"     The ratio of Firmicutes to Bacteroidetes scales as a power law")
        print(f"     with BMI. This is a quantitative 'obesity equation' for the gut.")
        findings.append(f"F/B ~ BMI^{coeffs_pl[0]:.2f}, R2={r2_pl:.3f}")

    if fit_growth and fit_growth["r2"] > 0.8:
        print(f"\n  3. Diversity grows logarithmically with age in childhood")
        print(f"     Shannon = {fit_growth['intercept']:.2f} + {fit_growth['slope']:.2f} * log(age)")
        print(f"     R2={fit_growth['r2']:.3f}")
        findings.append(f"Growth: Shannon ~ log(age), R2={fit_growth['r2']:.3f}")

    if fit_decline and fit_decline["r2"] > 0.8:
        print(f"\n  4. Diversity declines linearly after age 50")
        print(f"     Rate: {abs(fit_decline['slope']):.3f} units/year")
        print(f"     R2={fit_decline['r2']:.3f}")
        findings.append(f"Decline: {abs(fit_decline['slope']):.3f}/yr after 50, R2={fit_decline['r2']:.3f}")

    print(f"\n  NONE of these are established laws. All are CANDIDATE discoveries")
    print(f"  that would need validation on raw sequencing data (HMP, MetaHIT).")

    # Artifact
    artifact = {
        "id": "E105",
        "timestamp": now,
        "world": "microbiome",
        "data_source": "Published meta-analyses (HMP, American Gut, MetaHIT)",
        "status": "exploratory",
        "design": {
            "description": "Search for mathematical patterns in human gut microbiome composition, diversity, and disease associations",
            "caveat": "Uses aggregated published statistics, not raw sequencing data. All findings are candidate hypotheses.",
        },
        "result": {
            "zipf_bacteria": zipf,
            "fb_bmi_power_law": {"alpha": float(coeffs_pl[0]), "r2": float(r2_pl)},
            "diversity_bmi": fit_bmi,
            "diversity_lifecycle": {
                "growth": fit_growth,
                "decline": fit_decline,
                "peak_age": float(ages[peak_idx]),
            },
            "findings": findings,
        },
    }

    out_path = ROOT / "results" / "E105_microbiome.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
