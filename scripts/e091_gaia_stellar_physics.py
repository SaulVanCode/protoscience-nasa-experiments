#!/usr/bin/env python3
"""
E091 — Rediscovering Stellar Physics from ESA Gaia DR3

Question: Can we rediscover fundamental stellar physics laws from
the Gaia catalog alone — no textbook, no hints?

Data: ESA Gaia DR3 (1.8 billion stars, public archive)
  - Parallax (distance), temperature, luminosity, radius
  - Colors (BP-RP), magnitudes, surface gravity

Expected discoveries:
  - Hertzsprung-Russell diagram (main sequence)
  - Stefan-Boltzmann law: L = 4*pi*R^2*sigma*T^4  =>  L ~ R^2 * T^4
  - Mass-luminosity relation: L ~ M^3.5 (main sequence)
  - Distance-parallax: d = 1000/parallax (trivial but good sanity check)

Source: https://gea.esac.esa.int/archive/
"""

import json
import urllib.request
import urllib.parse
import csv
import io
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "gaia_dr3_stars.csv"

# ── Gaia TAP query ──────────────────────────────────────────────────
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"

# Get stars with good parallax, temperature, luminosity, and radius
ADQL_QUERY = """
SELECT TOP 15000
    g.source_id,
    g.ra, g.dec,
    g.parallax, g.parallax_error,
    g.parallax_over_error,
    g.phot_g_mean_mag,
    g.bp_rp,
    g.teff_gspphot,
    g.logg_gspphot,
    ap.lum_flame,
    ap.radius_flame,
    ap.mass_flame,
    ap.age_flame
FROM gaiadr3.gaia_source AS g
    JOIN gaiadr3.astrophysical_parameters AS ap ON g.source_id = ap.source_id
WHERE g.parallax_over_error > 10
    AND g.teff_gspphot IS NOT NULL
    AND ap.lum_flame IS NOT NULL
    AND ap.radius_flame IS NOT NULL
    AND ap.mass_flame IS NOT NULL
    AND g.phot_g_mean_mag < 14
    AND g.bp_rp IS NOT NULL
    AND ap.lum_flame > 0.01
    AND ap.lum_flame < 100000
    AND ap.radius_flame > 0.1
    AND ap.radius_flame < 500
    AND ap.mass_flame > 0.3
    AND ap.mass_flame < 20
ORDER BY g.random_index
"""


def fetch_gaia_data() -> list[dict]:
    """Fetch stellar data from ESA Gaia DR3 via TAP/ADQL."""
    if CACHE_FILE.exists():
        print(f"  Loading cached data from {CACHE_FILE}")
        records = []
        with open(CACHE_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = {}
                for k, v in row.items():
                    try:
                        rec[k] = float(v) if v else None
                    except ValueError:
                        rec[k] = v
                records.append(rec)
        return records

    print("  Querying ESA Gaia DR3 archive (this may take 30-60 seconds)...")
    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": ADQL_QUERY.strip(),
    }
    url = GAIA_TAP + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "ProtoScience/1.0 (stellar physics experiment)")

    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")

    # Parse CSV
    reader = csv.DictReader(io.StringIO(raw))
    records = []
    for row in reader:
        rec = {}
        for k, v in row.items():
            try:
                rec[k] = float(v) if v.strip() else None
            except (ValueError, AttributeError):
                rec[k] = v
        records.append(rec)

    # Cache
    if records:
        with open(CACHE_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"  Cached {len(records)} stars to {CACHE_FILE}")

    return records


def clean_data(records: list[dict]) -> dict:
    """Extract clean numpy arrays from records."""
    fields = ["teff_gspphot", "lum_flame", "radius_flame", "mass_flame",
              "parallax", "distance_gspphot", "logg_gspphot", "bp_rp",
              "phot_g_mean_mag", "mh_gspphot", "age_flame"]

    arrays = {}
    # Build mask: all key fields must be valid
    n = len(records)
    mask = np.ones(n, dtype=bool)
    for f in ["teff_gspphot", "lum_flame", "radius_flame", "mass_flame", "parallax"]:
        vals = []
        for r in records:
            v = r.get(f)
            vals.append(v if v is not None else np.nan)
        arr = np.array(vals, dtype=float)
        mask &= np.isfinite(arr) & (arr > 0)

    for f in fields:
        vals = []
        for r in records:
            v = r.get(f)
            vals.append(v if v is not None else np.nan)
        arr = np.array(vals, dtype=float)
        arrays[f] = arr[mask]

    # Compute absolute magnitude from parallax and apparent mag
    g_mag = arrays["phot_g_mean_mag"]
    plx = arrays["parallax"]  # milliarcseconds
    dist_pc = 1000.0 / plx  # parsecs
    abs_mag = g_mag - 5 * np.log10(np.clip(dist_pc, 1e-10, None)) + 5
    arrays["abs_mag_G"] = abs_mag
    arrays["dist_pc"] = dist_pc

    # Luminosity in solar units is already lum_flame
    # Temperature is teff_gspphot (K)
    # Radius is radius_flame (solar radii)
    # Mass is mass_flame (solar masses)

    print(f"  Clean dataset: {mask.sum()} stars (from {n} raw records)")
    return arrays


# ── Power-law fitting ───────────────────────────────────────────────

def fit_power_law(x, y, name_x="x", name_y="y"):
    """Fit y = C * x^alpha via log-log regression."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    if len(lx) < 10:
        return None

    coeffs = np.polyfit(lx, ly, 1)
    alpha = coeffs[0]
    C = 10 ** coeffs[1]

    y_pred = coeffs[0] * lx + coeffs[1]
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "relation": f"{name_y} = {C:.4f} * {name_x}^{alpha:.4f}",
        "alpha": float(alpha),
        "C": float(C),
        "r2": float(r2),
        "n": int(mask.sum()),
    }


def fit_multi_power(y, xs: dict):
    """Fit log(y) = sum(alpha_i * log(x_i)) + const via multivariate linear regression."""
    mask = np.isfinite(y) & (y > 0)
    for arr in xs.values():
        mask &= np.isfinite(arr) & (arr > 0)

    ly = np.log10(y[mask])
    X = np.column_stack([np.log10(xs[k][mask]) for k in xs] + [np.ones(mask.sum())])
    names = list(xs.keys())

    # Least squares
    result = np.linalg.lstsq(X, ly, rcond=None)
    coeffs = result[0]

    pred = X @ coeffs
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    terms = {names[i]: float(coeffs[i]) for i in range(len(names))}
    terms["const"] = float(10 ** coeffs[-1])

    relation_parts = " * ".join(f"{k}^{v:.3f}" for k, v in terms.items() if k != "const")
    relation = f"y = {terms['const']:.4f} * {relation_parts}"

    return {
        "relation": relation,
        "exponents": terms,
        "r2": float(r2),
        "n": int(mask.sum()),
    }


# ── Known laws to verify ───────────────────────────────────────────

KNOWN_LAWS = {
    "Stefan-Boltzmann: L ~ R^2 * T^4": {
        "expected_R": 2.0,
        "expected_T": 4.0,
    },
    "Mass-Luminosity: L ~ M^3.5": {
        "expected_alpha": 3.5,
        "tolerance": 0.5,  # broad because it varies by mass range
    },
}


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E091 -- Rediscovering Stellar Physics from ESA Gaia DR3")
    print("=" * 70)

    # 1. Fetch data
    print("\n  [1] Fetching Gaia DR3 data...")
    records = fetch_gaia_data()
    print(f"  Got {len(records)} records")

    # 2. Clean
    print("\n  [2] Cleaning and extracting arrays...")
    d = clean_data(records)

    T = d["teff_gspphot"]     # Temperature (K)
    L = d["lum_flame"]        # Luminosity (solar)
    R = d["radius_flame"]     # Radius (solar radii)
    M = d["mass_flame"]       # Mass (solar masses)
    bp_rp = d["bp_rp"]        # Color index
    abs_mag = d["abs_mag_G"]  # Absolute magnitude
    logg = d["logg_gspphot"]  # Surface gravity

    print(f"  T range: {np.nanmin(T):.0f} - {np.nanmax(T):.0f} K")
    print(f"  L range: {np.nanmin(L):.4f} - {np.nanmax(L):.1f} L_sun")
    print(f"  R range: {np.nanmin(R):.3f} - {np.nanmax(R):.1f} R_sun")
    print(f"  M range: {np.nanmin(M):.3f} - {np.nanmax(M):.1f} M_sun")

    # 3. Single power laws
    print("\n  [3] Power-law discovery (all pairs)...")
    pairs = [
        ("teff_gspphot", T, "lum_flame", L),
        ("mass_flame", M, "lum_flame", L),
        ("radius_flame", R, "lum_flame", L),
        ("teff_gspphot", T, "radius_flame", R),
        ("mass_flame", M, "radius_flame", R),
        ("mass_flame", M, "teff_gspphot", T),
        ("bp_rp", bp_rp, "abs_mag_G", abs_mag),
    ]

    discoveries = []
    for nx, x, ny, y in pairs:
        fit = fit_power_law(x, y, nx, ny)
        if fit:
            discoveries.append(fit)
            r2_str = f"{fit['r2']:.4f}"
            print(f"    {fit['relation']:60s}  R2={r2_str}  n={fit['n']}")

    # 4. Stefan-Boltzmann: L ~ R^2 * T^4
    print("\n  [4] Stefan-Boltzmann test: L = const * R^a * T^b")
    sb_fit = fit_multi_power(L, {"R": R, "T": T})
    print(f"    Result: {sb_fit['relation']}")
    print(f"    R2 = {sb_fit['r2']:.6f}  n = {sb_fit['n']}")
    print(f"    Expected exponents: R=2.0, T=4.0")
    print(f"    Found exponents:    R={sb_fit['exponents']['R']:.4f}, T={sb_fit['exponents']['T']:.4f}")

    R_err = abs(sb_fit["exponents"]["R"] - 2.0)
    T_err = abs(sb_fit["exponents"]["T"] - 4.0)
    sb_verdict = "REDISCOVERED" if R_err < 0.2 and T_err < 0.5 else "PARTIAL"
    print(f"    Verdict: {sb_verdict}")

    # 5. Mass-Luminosity (main sequence only: logg > 3.5)
    print("\n  [5] Mass-Luminosity relation (main sequence filter: log(g) > 3.5)")
    ms_mask = logg > 3.5
    ml_fit = fit_power_law(M[ms_mask], L[ms_mask], "M", "L")
    if ml_fit:
        print(f"    {ml_fit['relation']}")
        print(f"    R2 = {ml_fit['r2']:.4f}  n = {ml_fit['n']}")
        print(f"    Expected: L ~ M^3.5  (varies 3.0-4.0 by mass range)")
        print(f"    Found:    L ~ M^{ml_fit['alpha']:.4f}")
        ml_verdict = "REDISCOVERED" if abs(ml_fit["alpha"] - 3.5) < 1.0 else "PARTIAL"
        print(f"    Verdict: {ml_verdict}")
    else:
        ml_fit = {"alpha": 0, "r2": 0, "n": 0, "relation": "FAILED"}
        ml_verdict = "FAILED"

    # 6. Mass-Luminosity by mass bin
    print("\n  [6] Mass-Luminosity by mass range:")
    mass_bins = [(0.3, 0.8, "low-mass"), (0.8, 2.0, "solar-type"), (2.0, 20.0, "high-mass")]
    ml_by_bin = []
    for mlo, mhi, label in mass_bins:
        bm = ms_mask & (M >= mlo) & (M < mhi)
        if bm.sum() > 20:
            bf = fit_power_law(M[bm], L[bm], "M", "L")
            if bf:
                ml_by_bin.append({"range": label, "m_lo": mlo, "m_hi": mhi, **bf})
                print(f"    {label:12s} ({mlo}-{mhi} M_sun): L ~ M^{bf['alpha']:.3f}  R2={bf['r2']:.4f}  n={bf['n']}")

    # 7. HR diagram statistics
    print("\n  [7] Hertzsprung-Russell diagram structure:")
    # Main sequence fraction
    ms_count = ms_mask.sum()
    total = len(logg)
    print(f"    Main sequence (log g > 3.5): {ms_count} / {total} ({100*ms_count/total:.1f}%)")
    # Giants
    giant_mask = (logg > 1.0) & (logg <= 3.5)
    print(f"    Giants (1.0 < log g <= 3.5): {giant_mask.sum()} ({100*giant_mask.sum()/total:.1f}%)")
    # Supergiants
    sg_mask = logg <= 1.0
    print(f"    Supergiants (log g <= 1.0):  {sg_mask.sum()} ({100*sg_mask.sum()/total:.1f}%)")

    # Color-magnitude correlation
    cm_mask = np.isfinite(bp_rp) & np.isfinite(abs_mag)
    if cm_mask.sum() > 10:
        r_cm = np.corrcoef(bp_rp[cm_mask], abs_mag[cm_mask])[0, 1]
        print(f"    Color-magnitude correlation (BP-RP vs M_G): r = {r_cm:.4f}")

    # 8. Summary
    print("\n  " + "=" * 60)
    print("  SUMMARY OF DISCOVERIES")
    print("  " + "=" * 60)

    n_rediscovered = 0
    results_summary = {}

    # Stefan-Boltzmann
    print(f"\n  Stefan-Boltzmann (L ~ R^2 * T^4):")
    print(f"    Found: R={sb_fit['exponents']['R']:.3f}, T={sb_fit['exponents']['T']:.3f}")
    print(f"    R2 = {sb_fit['r2']:.4f}")
    print(f"    [{sb_verdict}]")
    if sb_verdict == "REDISCOVERED":
        n_rediscovered += 1
    results_summary["stefan_boltzmann"] = {
        "verdict": sb_verdict,
        "R_exp": sb_fit["exponents"]["R"],
        "T_exp": sb_fit["exponents"]["T"],
        "r2": sb_fit["r2"],
    }

    # Mass-Luminosity
    print(f"\n  Mass-Luminosity (L ~ M^3.5):")
    print(f"    Found: M^{ml_fit['alpha']:.3f}")
    print(f"    R2 = {ml_fit['r2']:.4f}")
    print(f"    [{ml_verdict}]")
    if ml_verdict == "REDISCOVERED":
        n_rediscovered += 1
    results_summary["mass_luminosity"] = {
        "verdict": ml_verdict,
        "alpha": ml_fit["alpha"],
        "r2": ml_fit["r2"],
    }

    print(f"\n  Score: {n_rediscovered}/2 fundamental laws rediscovered")
    print(f"  Data source: ESA Gaia DR3 (gea.esac.esa.int)")

    # 9. Artifact
    artifact = {
        "id": "E091",
        "timestamp": now,
        "world": "gaia_stars",
        "data_source": "ESA Gaia DR3",
        "data_url": "https://gea.esac.esa.int/archive/",
        "status": "passed" if n_rediscovered >= 1 else "partial",
        "design": {
            "description": "Fetch stellar parameters from ESA Gaia DR3, discover power-law relations, verify Stefan-Boltzmann and mass-luminosity laws",
            "query": ADQL_QUERY.strip(),
            "n_stars_requested": 15000,
            "filters": "parallax_over_error > 10, G < 14, valid T/L/R/M",
        },
        "result": {
            "n_stars_clean": int(len(T)),
            "power_law_discoveries": discoveries,
            "stefan_boltzmann": results_summary["stefan_boltzmann"],
            "mass_luminosity": results_summary["mass_luminosity"],
            "mass_luminosity_by_bin": ml_by_bin,
            "hr_structure": {
                "main_sequence_frac": float(ms_count / total),
                "giant_frac": float(giant_mask.sum() / total),
            },
            "n_rediscovered": n_rediscovered,
        },
    }

    out_path = ROOT / "results" / "E091_gaia_stellar_physics.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
