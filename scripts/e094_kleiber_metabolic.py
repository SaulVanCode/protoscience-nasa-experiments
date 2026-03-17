#!/usr/bin/env python3
"""
E094 — Rediscovering Kleiber's Law from Mammalian Data

Question: Can we rediscover the 3/4 power law of metabolic scaling
from raw species data?

Data: PanTHERIA database (5,416 mammal species)
  - Body mass, basal metabolic rate, longevity, litter size, etc.

Expected discoveries:
  - Kleiber's Law: BMR ~ M^0.75 (basal metabolic rate scales with mass^3/4)
  - Longevity ~ M^0.25 (lifespan scales with mass^1/4)
  - Heart rate ~ M^(-0.25)
  - Gestation ~ M^0.25
  - "Quarter-power scaling" universality

Source: PanTHERIA (Ecological Archives E090-184)
  Jones et al. (2009) Ecology 90:2648
"""

import json
import urllib.request
import csv
import io
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "pantheria_mammals.txt"

PANTHERIA_URL = "https://esapubs.org/archive/ecol/E090/184/PanTHERIA_1-0_WR05_Aug2008.txt"

MISSING = -999.0


def fetch_data() -> list[dict]:
    """Fetch PanTHERIA mammal database."""
    if CACHE_FILE.exists():
        print(f"  Loading cached data from {CACHE_FILE}")
        raw = CACHE_FILE.read_text(encoding="utf-8")
    else:
        print("  Fetching PanTHERIA database...")
        req = urllib.request.Request(PANTHERIA_URL)
        req.add_header("User-Agent", "ProtoScience/1.0 (metabolic scaling experiment)")
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        CACHE_FILE.write_text(raw, encoding="utf-8")
        print(f"  Cached to {CACHE_FILE}")

    # Tab-separated
    reader = csv.DictReader(io.StringIO(raw), delimiter="\t")
    records = list(reader)
    print(f"  Got {len(records)} species")
    return records


def extract_float(record: dict, key: str) -> float:
    """Extract float value, return NaN for missing."""
    val = record.get(key, "").strip()
    try:
        v = float(val)
        return np.nan if v == MISSING else v
    except (ValueError, TypeError):
        return np.nan


def clean_data(records: list[dict]) -> dict:
    """Extract arrays for key variables."""
    fields = {
        "body_mass_g": "5-1_AdultBodyMass_g",
        "bmr_mlo2hr": "18-1_BasalMetRate_mLO2hr",
        "longevity_mo": "17-1_MaxLongevity_m",
        "gestation_d": "9-1_GestationLen_d",
        "litter_size": "15-1_LitterSize",
        "neonate_mass_g": "5-3_NeonateBodyMass_g",
        "weaning_age_d": "25-1_WeaningAge_d",
        "home_range_km2": "22-1_HomeRange_km2",
        "pop_density_n_km2": "21-1_PopulationDensity_n/km2",
    }

    arrays = {}
    for name, col in fields.items():
        vals = [extract_float(r, col) for r in records]
        arrays[name] = np.array(vals)

    return arrays


def fit_power_law(x, y, name_x="x", name_y="y"):
    """Fit y = C * x^alpha via log-log regression."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 10:
        return None
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    coeffs = np.polyfit(lx, ly, 1)
    alpha, C = coeffs[0], 10 ** coeffs[1]

    pred = np.polyval(coeffs, lx)
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "relation": f"{name_y} = {C:.4f} * {name_x}^{alpha:.4f}",
        "alpha": float(alpha),
        "C": float(C),
        "r2": float(r2),
        "n": int(mask.sum()),
    }


KNOWN_LAWS = {
    "Kleiber's Law: BMR ~ M^0.75": {
        "x": "body_mass_g", "y": "bmr_mlo2hr",
        "expected_alpha": 0.75, "tolerance": 0.1,
    },
    "Longevity ~ M^0.25": {
        "x": "body_mass_g", "y": "longevity_mo",
        "expected_alpha": 0.25, "tolerance": 0.1,
    },
    "Gestation ~ M^0.25": {
        "x": "body_mass_g", "y": "gestation_d",
        "expected_alpha": 0.25, "tolerance": 0.1,
    },
    "Litter size ~ M^(-0.25)": {
        "x": "body_mass_g", "y": "litter_size",
        "expected_alpha": -0.25, "tolerance": 0.15,
    },
    "Neonate mass ~ M^0.75": {
        "x": "body_mass_g", "y": "neonate_mass_g",
        "expected_alpha": 0.75, "tolerance": 0.15,
    },
    "Home range ~ M^1.0": {
        "x": "body_mass_g", "y": "home_range_km2",
        "expected_alpha": 1.0, "tolerance": 0.2,
    },
    "Pop density ~ M^(-0.75)": {
        "x": "body_mass_g", "y": "pop_density_n_km2",
        "expected_alpha": -0.75, "tolerance": 0.2,
    },
}


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E094 -- Rediscovering Kleiber's Law from Mammalian Data")
    print("=" * 70)

    # 1. Fetch
    print("\n  [1] Fetching PanTHERIA database...")
    records = fetch_data()

    # 2. Clean
    print("\n  [2] Extracting arrays...")
    d = clean_data(records)

    for name, arr in d.items():
        valid = np.sum(np.isfinite(arr) & (arr > 0))
        print(f"    {name:25s}: {valid:5d} valid values")

    # 3. All power-law pairs against body mass
    print("\n  [3] Power-law discovery: everything vs body mass")
    discoveries = []
    for name, arr in d.items():
        if name == "body_mass_g":
            continue
        fit = fit_power_law(d["body_mass_g"], arr, "M", name)
        if fit and fit["r2"] > 0.05:
            discoveries.append(fit)
            print(f"    {fit['relation']:55s}  R2={fit['r2']:.4f}  n={fit['n']}")

    # 4. Verify known laws
    print("\n  [4] Verification against known biological scaling laws:")
    n_rediscovered = 0
    verification = {}

    for law_name, spec in KNOWN_LAWS.items():
        x = d[spec["x"]]
        y = d[spec["y"]]
        fit = fit_power_law(x, y, "M", spec["y"])

        if fit is None:
            verification[law_name] = {"verdict": "NO_DATA"}
            print(f"    [--] {law_name}: insufficient data")
            continue

        err = abs(fit["alpha"] - spec["expected_alpha"])
        verdict = "REDISCOVERED" if err < spec["tolerance"] else "PARTIAL"
        if verdict == "REDISCOVERED":
            n_rediscovered += 1

        verification[law_name] = {
            "verdict": verdict,
            "expected_alpha": spec["expected_alpha"],
            "found_alpha": fit["alpha"],
            "error": err,
            "r2": fit["r2"],
            "n": fit["n"],
        }
        print(f"    [{'OK' if verdict == 'REDISCOVERED' else '~~'}] {law_name}")
        print(f"        Found: M^{fit['alpha']:.4f} (expected {spec['expected_alpha']:.2f}, err={err:.4f})  R2={fit['r2']:.4f}  n={fit['n']}")

    total = len(KNOWN_LAWS)
    print(f"\n  Score: {n_rediscovered}/{total} scaling laws rediscovered")

    # 5. Quarter-power universality check
    print("\n  [5] Quarter-power universality:")
    quarter_powers = []
    for name, v in verification.items():
        if v.get("verdict") in ("REDISCOVERED", "PARTIAL") and "found_alpha" in v:
            alpha = v["found_alpha"]
            # Check if alpha is close to n/4 for some integer n
            nearest_quarter = round(alpha * 4) / 4
            quarter_powers.append({
                "law": name,
                "alpha": alpha,
                "nearest_quarter": nearest_quarter,
                "deviation": abs(alpha - nearest_quarter),
            })
            print(f"    {name:35s}  alpha={alpha:.4f}  nearest n/4={nearest_quarter:.2f}  dev={abs(alpha - nearest_quarter):.4f}")

    # Summary
    print("\n  " + "=" * 60)
    print("  SUMMARY")
    print("  " + "=" * 60)
    print(f"\n  {n_rediscovered}/{total} biological scaling laws rediscovered")
    print(f"  Data: PanTHERIA ({len(records)} mammal species)")
    print(f"  Source: Jones et al. (2009) Ecology 90:2648")

    # Artifact
    artifact = {
        "id": "E094",
        "timestamp": now,
        "world": "biology",
        "data_source": "PanTHERIA (Ecological Archives E090-184)",
        "data_url": "https://esapubs.org/archive/ecol/E090/184/",
        "status": "passed" if n_rediscovered >= 3 else "partial",
        "design": {
            "description": "Fetch PanTHERIA mammal database, discover allometric scaling laws (Kleiber's Law and quarter-power scaling)",
            "n_species": len(records),
        },
        "result": {
            "power_law_discoveries": discoveries,
            "verification": verification,
            "n_rediscovered": n_rediscovered,
            "n_total": total,
            "quarter_power_analysis": quarter_powers,
        },
    }

    out_path = ROOT / "results" / "E094_kleiber_metabolic.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
