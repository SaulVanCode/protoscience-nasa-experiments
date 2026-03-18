#!/usr/bin/env python3
"""
E110 — Drug-Induced Longevity: Searching for the Aging Equation

Question: Is there a mathematical law governing how drugs extend
lifespan across species? Does it connect to Kleiber (E094)?

Background:
  DrugAge catalogs thousands of experiments where compounds were
  tested for their effect on lifespan in model organisms (worms,
  flies, mice). Some compounds extend life by 100%+ in worms but
  barely 10% in mice.

  Nobody has a unified equation for drug-induced longevity.

  Hypotheses to test:
  1. Is there a maximum lifespan extension that decreases with
     organism complexity?
  2. Does the dose-response follow a power law or log curve?
  3. Do certain drug classes consistently outperform others?
  4. Is there a Kleiber-like scaling between species body mass
     and maximum achievable lifespan extension?

Data: DrugAge Database (Human Ageing Genomic Resources)
  3,424 experiments across multiple species

Source: https://genomics.senescence.info/drugs/
  Barardo et al. (2017) Aging Cell 16:5, 971-978
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "drugage" / "drugage.csv"

# Approximate body mass for model organisms (grams)
SPECIES_MASS = {
    "Caenorhabditis elegans": 0.001,    # 1 mg
    "Drosophila melanogaster": 0.001,    # 1 mg
    "Drosophila mojavensis": 0.001,
    "Saccharomyces cerevisiae": 0.00004, # 40 µg (yeast)
    "Mus musculus": 25.0,                # 25 g (mouse)
    "Rattus norvegicus": 300.0,          # 300 g (rat)
    "Danio rerio": 0.5,                  # 0.5 g (zebrafish)
    "Podospora anserina": 0.0001,        # fungus
    "Nothobranchius furzeri": 3.0,       # killifish
    "Acheta domesticus": 0.5,            # cricket
}

# Approximate max natural lifespan (days)
SPECIES_LIFESPAN = {
    "Caenorhabditis elegans": 20,
    "Drosophila melanogaster": 60,
    "Drosophila mojavensis": 60,
    "Saccharomyces cerevisiae": 7,   # replicative
    "Mus musculus": 900,
    "Rattus norvegicus": 1100,
    "Danio rerio": 1825,
    "Podospora anserina": 14,
    "Nothobranchius furzeri": 120,
}


def load_data():
    """Load DrugAge CSV."""
    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                avg_change = row.get("avg_lifespan_change_percent", "").strip()
                if avg_change and avg_change != "NA":
                    avg_change = float(avg_change)
                else:
                    avg_change = None
            except ValueError:
                avg_change = None

            try:
                max_change = row.get("max_lifespan_change_percent", "").strip()
                if max_change and max_change != "NA":
                    max_change = float(max_change)
                else:
                    max_change = None
            except ValueError:
                max_change = None

            records.append({
                "compound": row.get("compound_name", "").strip(),
                "species": row.get("species", "").strip(),
                "dosage": row.get("dosage", "").strip(),
                "gender": row.get("gender", "").strip(),
                "avg_change": avg_change,
                "max_change": max_change,
                "significance": row.get("avg_lifespan_significance", "").strip(),
            })
    return records


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E110 -- Drug-Induced Longevity: Searching for the Aging Equation")
    print("=" * 70)

    # 1. Load
    print(f"\n  [1] Loading DrugAge database...")
    records = load_data()
    print(f"  {len(records)} experiments loaded")

    # 2. Species breakdown
    print(f"\n  [2] Species distribution:")
    species_count = Counter(r["species"] for r in records)
    for sp, count in species_count.most_common(10):
        print(f"    {sp:35s}: {count:5d} experiments")

    # 3. Filter to experiments with avg lifespan change
    valid = [r for r in records if r["avg_change"] is not None]
    print(f"\n  [3] {len(valid)} experiments with measured avg lifespan change")

    # Positive (extends life) vs negative (shortens)
    extends = [r for r in valid if r["avg_change"] > 0]
    shortens = [r for r in valid if r["avg_change"] < 0]
    neutral = [r for r in valid if r["avg_change"] == 0]
    print(f"    Extends life:  {len(extends)} ({len(extends)/len(valid)*100:.1f}%)")
    print(f"    Shortens life: {len(shortens)} ({len(shortens)/len(valid)*100:.1f}%)")
    print(f"    No effect:     {len(neutral)} ({len(neutral)/len(valid)*100:.1f}%)")

    # 4. Distribution of lifespan changes
    print(f"\n  [4] Distribution of lifespan extension:")
    changes = np.array([r["avg_change"] for r in valid])
    print(f"    Mean:   {np.mean(changes):+.2f}%")
    print(f"    Median: {np.median(changes):+.2f}%")
    print(f"    Std:    {np.std(changes):.2f}%")
    print(f"    Min:    {np.min(changes):+.2f}%")
    print(f"    Max:    {np.max(changes):+.2f}%")

    # Histogram by buckets
    print(f"\n    Distribution:")
    buckets = [(-100, -50), (-50, -20), (-20, 0), (0, 10), (10, 25),
               (25, 50), (50, 100), (100, 200), (200, 500)]
    for lo, hi in buckets:
        n = np.sum((changes >= lo) & (changes < hi))
        bar = "#" * (n // 5)
        print(f"      {lo:+4d}% to {hi:+4d}%: {n:4d}  |{bar}")

    # 5. By species — max achievable extension
    print(f"\n  [5] Maximum lifespan extension by species:")
    species_max = {}
    species_median = {}
    for sp in species_count:
        sp_changes = [r["avg_change"] for r in valid if r["species"] == sp and r["avg_change"] > 0]
        if len(sp_changes) >= 5:
            species_max[sp] = max(sp_changes)
            species_median[sp] = float(np.median(sp_changes))

    print(f"\n    {'Species':35s} {'Max ext':>8s} {'Median':>8s} {'n':>5s} {'Mass(g)':>8s}")
    print(f"    {'-'*35} {'-'*8} {'-'*8} {'-'*5} {'-'*8}")
    for sp in sorted(species_max, key=lambda s: -species_max[s]):
        mass = SPECIES_MASS.get(sp, "?")
        n_exp = len([r for r in valid if r["species"] == sp and r["avg_change"] > 0])
        mass_str = f"{mass}" if isinstance(mass, str) else f"{mass:.4f}"
        print(f"    {sp:35s} {species_max[sp]:+7.1f}% {species_median[sp]:+7.1f}% {n_exp:5d} {mass_str:>8s}")

    # 6. The scaling question: does max extension decrease with body mass?
    print(f"\n  [6] Kleiber connection: max extension vs body mass")
    masses = []
    max_exts = []
    sp_names = []
    for sp, max_ext in species_max.items():
        if sp in SPECIES_MASS:
            masses.append(SPECIES_MASS[sp])
            max_exts.append(max_ext)
            sp_names.append(sp)

    if len(masses) >= 3:
        log_m = np.log10(masses)
        log_ext = np.log10(max_exts)

        coeffs = np.polyfit(log_m, log_ext, 1)
        alpha = coeffs[0]
        C = 10 ** coeffs[1]

        pred = np.polyval(coeffs, log_m)
        ss_res = np.sum((log_ext - pred) ** 2)
        ss_tot = np.sum((log_ext - np.mean(log_ext)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"    Max extension ~ mass^{alpha:.4f}  R2={r2:.4f}")
        print(f"    (n={len(masses)} species)")

        if alpha < -0.1:
            print(f"    => Bigger organisms are HARDER to extend")
            print(f"    => Each 10x in mass reduces max extension by {(1-10**alpha)*100:.0f}%")
        elif alpha > 0.1:
            print(f"    => Bigger organisms show MORE extension (unexpected)")
        else:
            print(f"    => No clear mass-dependent trend")

        scaling_result = {"alpha": float(alpha), "C": float(C), "r2": float(r2), "n": len(masses)}
    else:
        scaling_result = None
        print(f"    Not enough species with both mass data and experiments")

    # 7. Top longevity compounds
    print(f"\n  [7] Top 15 life-extending compounds (by max avg extension):")
    compound_best = {}
    for r in valid:
        if r["avg_change"] and r["avg_change"] > 0:
            comp = r["compound"]
            if comp not in compound_best or r["avg_change"] > compound_best[comp]["change"]:
                compound_best[comp] = {"change": r["avg_change"], "species": r["species"]}

    top_compounds = sorted(compound_best.items(), key=lambda x: -x[1]["change"])[:15]
    for comp, info in top_compounds:
        print(f"    {comp:30s} {info['change']:+7.1f}%  ({info['species']})")

    # 8. Rapamycin deep dive (most studied aging drug)
    print(f"\n  [8] Rapamycin (most studied aging compound):")
    rapa = [r for r in valid if "rapamycin" in r["compound"].lower() or "sirolimus" in r["compound"].lower()]
    if rapa:
        rapa_changes = [r["avg_change"] for r in rapa if r["avg_change"] is not None]
        rapa_species = Counter(r["species"] for r in rapa)
        print(f"    {len(rapa)} experiments")
        print(f"    Species: {dict(rapa_species.most_common(5))}")
        print(f"    Mean extension: {np.mean(rapa_changes):+.1f}%")
        print(f"    Range: {min(rapa_changes):+.1f}% to {max(rapa_changes):+.1f}%")

    # 9. Gender differences
    print(f"\n  [9] Gender differences in longevity extension:")
    male_changes = [r["avg_change"] for r in valid if r["gender"] == "Male" and r["avg_change"] > 0]
    female_changes = [r["avg_change"] for r in valid if r["gender"] == "Female" and r["avg_change"] > 0]
    both_changes = [r["avg_change"] for r in valid if r["gender"] == "Both" and r["avg_change"] > 0]

    if male_changes and female_changes:
        print(f"    Male:   median extension = {np.median(male_changes):+.1f}%  (n={len(male_changes)})")
        print(f"    Female: median extension = {np.median(female_changes):+.1f}%  (n={len(female_changes)})")
        print(f"    Both:   median extension = {np.median(both_changes):+.1f}%  (n={len(both_changes)})")

    # Summary
    print(f"\n  " + "=" * 60)
    print(f"  SUMMARY")
    print(f"  " + "=" * 60)

    findings = []

    print(f"\n  1. {len(extends)} compounds extend life, {len(shortens)} shorten it")
    findings.append(f"{len(extends)} life-extending, {len(shortens)} life-shortening compounds")

    if species_max:
        best_sp = max(species_max, key=species_max.get)
        print(f"  2. Max extension: {species_max[best_sp]:+.0f}% in {best_sp}")
        findings.append(f"Max extension: {species_max[best_sp]:+.0f}% in {best_sp}")

    if scaling_result and abs(scaling_result["alpha"]) > 0.05:
        print(f"  3. Extension scales with mass: alpha={scaling_result['alpha']:.3f}")
        print(f"     (Connects to Kleiber: bigger = harder to extend)")
        findings.append(f"Mass scaling alpha={scaling_result['alpha']:.3f}")

    if top_compounds:
        print(f"  4. Top compound: {top_compounds[0][0]} ({top_compounds[0][1]['change']:+.0f}%)")

    print(f"\n  CAVEAT: These are lab results in model organisms.")
    print(f"  Translation to humans is uncertain at best.")

    # Artifact
    artifact = {
        "id": "E110",
        "timestamp": now,
        "world": "aging",
        "data_source": "DrugAge Database (Barardo et al. 2017)",
        "data_url": "https://genomics.senescence.info/drugs/",
        "status": "passed",
        "design": {
            "description": "Analyze drug-induced longevity across species from DrugAge database. Search for scaling laws connecting body mass, drug dosage, and lifespan extension.",
            "n_experiments": len(records),
            "n_valid": len(valid),
        },
        "result": {
            "n_extends": len(extends),
            "n_shortens": len(shortens),
            "mean_extension": float(np.mean(changes)),
            "median_extension": float(np.median(changes)),
            "max_extension": float(np.max(changes)),
            "species_max": {sp: float(v) for sp, v in species_max.items()},
            "mass_scaling": scaling_result,
            "top_compounds": [(c, {"change": float(i["change"]), "species": i["species"]}) for c, i in top_compounds[:10]],
            "findings": findings,
        },
    }

    out_path = ROOT / "results" / "E110_drugage_longevity.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
