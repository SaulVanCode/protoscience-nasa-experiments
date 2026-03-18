#!/usr/bin/env python3
"""
E098 — Searching for Laws in Human Geography and Society

Question: Are there mathematical relationships between geography,
climate, economics, and human behavior across countries?

Data: World Bank Open Data API (200+ countries)
  - GDP per capita, life expectancy, suicide rate, temperature
  - Latitude, population density, education, CO2 emissions

This is explicitly NOISY social data. R² values will be low.
The goal is to see what ProtoScience finds — and what it doesn't.

Source: https://data.worldbank.org/ (World Bank Open Data)
"""

import json
import urllib.request
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_FILE = DATA_DIR / "world_bank_countries.json"

# World Bank indicators we want
INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita",          # GDP per capita (current US$)
    "SP.DYN.LE00.IN": "life_expectancy",          # Life expectancy at birth
    "SP.POP.TOTL": "population",                   # Total population
    "EN.ATM.CO2E.PC": "co2_per_capita",           # CO2 emissions per capita
    "SE.XPD.TOTL.GD.ZS": "education_pct_gdp",    # Education spending % GDP
    "SH.XPD.CHEX.GD.ZS": "health_pct_gdp",       # Health spending % GDP
    "SP.DYN.TFRT.IN": "fertility_rate",           # Fertility rate
    "SH.STA.SUIC.P5": "suicide_rate",             # Suicide mortality rate per 100k
    "AG.LND.ARBL.ZS": "arable_land_pct",          # Arable land % of land area
    "EN.POP.DNST": "pop_density",                  # Population density (per km²)
}

# Country coordinates (latitude) — manual for key countries
# We'll use this as a proxy for climate/temperature
COUNTRY_LAT = {
    "NOR": 60, "SWE": 62, "FIN": 64, "ISL": 65, "DNK": 56,
    "CAN": 56, "RUS": 60, "GBR": 54, "DEU": 51, "FRA": 46,
    "USA": 38, "JPN": 36, "KOR": 37, "CHN": 35, "IND": 20,
    "BRA": -10, "MEX": 23, "ARG": -34, "AUS": -25, "NZL": -41,
    "ZAF": -29, "NGA": 10, "EGY": 27, "KEN": 0, "ETH": 9,
    "COD": -4, "TZA": -6, "GHA": 8, "CIV": 7, "CMR": 6,
    "SEN": 14, "MLI": 17, "NER": 18, "TCD": 15, "SDN": 15,
    "MOZ": -18, "MDG": -20, "AGO": -12, "ZMB": -15, "ZWE": -20,
    "ESP": 40, "ITA": 42, "PRT": 39, "GRC": 38, "TUR": 39,
    "POL": 52, "UKR": 49, "ROU": 46, "HUN": 47, "CZE": 50,
    "AUT": 48, "CHE": 47, "BEL": 51, "NLD": 52, "IRL": 53,
    "SAU": 24, "ARE": 24, "QAT": 25, "KWT": 29, "IRQ": 33,
    "IRN": 32, "PAK": 30, "BGD": 24, "IDN": -5, "THA": 15,
    "VNM": 16, "PHL": 13, "MYS": 4, "SGP": 1, "MMR": 19,
    "COL": 4, "PER": -10, "CHL": -30, "VEN": 8, "ECU": -2,
    "BOL": -17, "PRY": -23, "URY": -33, "CRI": 10, "PAN": 9,
    "GTM": 15, "HND": 15, "SLV": 14, "NIC": 13, "DOM": 19,
    "HTI": 19, "JAM": 18, "CUB": 22, "LKA": 7, "NPL": 28,
    "ISR": 31, "JOR": 31, "LBN": 34, "OMN": 21, "BHR": 26,
    "LUX": 50, "EST": 59, "LVA": 57, "LTU": 56, "SVK": 49,
    "SVN": 46, "HRV": 45, "SRB": 44, "BGR": 43, "ALB": 41,
    "MKD": 41, "BIH": 44, "MNE": 43, "MDA": 47, "BLR": 54,
    "GEO": 42, "ARM": 40, "AZE": 41, "KAZ": 48, "UZB": 41,
    "TKM": 39, "KGZ": 41, "TJK": 39, "MNG": 48, "TWN": 24,
}

# Approximate mean annual temperature by latitude (simplified)
def lat_to_temp(lat):
    """Rough mean annual temperature from latitude."""
    return 27 - 0.5 * abs(lat)


def fetch_world_bank() -> dict:
    """Fetch country indicators from World Bank API."""
    if CACHE_FILE.exists():
        print(f"  Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE, "r") as f:
            return json.load(f)

    print("  Fetching World Bank indicators...")
    country_data = {}

    for indicator_id, name in INDICATORS.items():
        print(f"    Fetching {name}...")
        url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_id}?date=2020:2022&format=json&per_page=1000"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ProtoScience/1.0")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if len(data) < 2 or not data[1]:
                continue

            for record in data[1]:
                iso = record.get("countryiso3code", "")
                val = record.get("value")
                if not iso or val is None or len(iso) != 3:
                    continue

                if iso not in country_data:
                    country_data[iso] = {"iso": iso, "name": record["country"]["value"]}

                # Take most recent non-null value
                if name not in country_data[iso] or country_data[iso][name] is None:
                    country_data[iso][name] = float(val)

        except Exception as e:
            print(f"      Failed: {e}")

    # Add latitude and temperature
    for iso, lat in COUNTRY_LAT.items():
        if iso in country_data:
            country_data[iso]["latitude"] = lat
            country_data[iso]["abs_latitude"] = abs(lat)
            country_data[iso]["approx_temp"] = lat_to_temp(lat)

    with open(CACHE_FILE, "w") as f:
        json.dump(country_data, f, indent=2)
    print(f"  Cached {len(country_data)} countries")

    return country_data


def extract_pair(countries: dict, x_key: str, y_key: str):
    """Extract matched x, y arrays for countries that have both values."""
    x, y, names = [], [], []
    for iso, c in countries.items():
        xv = c.get(x_key)
        yv = c.get(y_key)
        if xv is not None and yv is not None and xv > 0 and yv > 0:
            x.append(xv)
            y.append(yv)
            names.append(c.get("name", iso))
    return np.array(x), np.array(y), names


def fit_linear(x, y):
    """Linear fit y = a + b*x."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return None
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(coeffs[0]), "intercept": float(coeffs[1]),
            "r2": float(r2), "n": int(len(x))}


def fit_log_log(x, y):
    """Power law fit via log-log."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return None
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    coeffs = np.polyfit(lx, ly, 1)
    alpha, C = coeffs[0], 10 ** coeffs[1]
    pred = np.polyval(coeffs, lx)
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha": float(alpha), "C": float(C), "r2": float(r2), "n": int(mask.sum())}


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E098 -- Laws in Human Geography and Society")
    print("=" * 70)

    # 1. Fetch
    print("\n  [1] Fetching World Bank data...")
    countries = fetch_world_bank()
    n_countries = len(countries)
    print(f"  {n_countries} countries/regions loaded")

    # Count countries with latitude
    n_with_lat = sum(1 for c in countries.values() if "latitude" in c)
    print(f"  {n_with_lat} countries with latitude data")

    # 2. Key relationships
    print("\n  [2] Searching for relationships...\n")

    tests = [
        ("GDP vs Life Expectancy", "gdp_per_capita", "life_expectancy", "log_log"),
        ("GDP vs Fertility", "gdp_per_capita", "fertility_rate", "log_log"),
        ("GDP vs CO2", "gdp_per_capita", "co2_per_capita", "log_log"),
        ("GDP vs Suicide Rate", "gdp_per_capita", "suicide_rate", "log_log"),
        ("Latitude vs GDP", "abs_latitude", "gdp_per_capita", "linear"),
        ("Latitude vs Life Expectancy", "abs_latitude", "life_expectancy", "linear"),
        ("Latitude vs Suicide Rate", "abs_latitude", "suicide_rate", "linear"),
        ("Temperature vs GDP", "approx_temp", "gdp_per_capita", "linear"),
        ("Temperature vs Fertility", "approx_temp", "fertility_rate", "linear"),
        ("Temperature vs Life Expectancy", "approx_temp", "life_expectancy", "linear"),
        ("Population Density vs GDP", "pop_density", "gdp_per_capita", "log_log"),
        ("Fertility vs Life Expectancy", "fertility_rate", "life_expectancy", "linear"),
        ("Education vs Life Expectancy", "education_pct_gdp", "life_expectancy", "linear"),
        ("Health Spending vs Life Expectancy", "health_pct_gdp", "life_expectancy", "linear"),
        ("CO2 vs Life Expectancy", "co2_per_capita", "life_expectancy", "log_log"),
    ]

    results = []
    for name, x_key, y_key, method in tests:
        x, y, _ = extract_pair(countries, x_key, y_key)
        if len(x) < 10:
            continue

        if method == "log_log":
            fit = fit_log_log(x, y)
            if fit:
                results.append({"test": name, "method": "power_law", **fit})
                tag = f"y ~ x^{fit['alpha']:.3f}"
        else:
            fit = fit_linear(x, y)
            if fit:
                results.append({"test": name, "method": "linear", **fit})
                tag = f"y = {fit['slope']:.4f}*x + {fit['intercept']:.1f}"

        if fit:
            r2_bar = "#" * int(fit["r2"] * 30)
            print(f"    {name:40s}  R2={fit['r2']:.4f} |{r2_bar:30s}|  n={fit['n']}  {tag}")

    # Sort by R²
    results.sort(key=lambda r: -r["r2"])

    # 3. Top findings
    print("\n  " + "=" * 60)
    print("  TOP FINDINGS (sorted by R²)")
    print("  " + "=" * 60)

    for i, r in enumerate(results[:10]):
        print(f"\n  {i+1}. {r['test']}")
        print(f"     R² = {r['r2']:.4f}  (n={r['n']})")
        if r["method"] == "power_law":
            print(f"     y = {r['C']:.4f} * x^{r['alpha']:.4f}")
        else:
            print(f"     y = {r['slope']:.4f}*x + {r['intercept']:.1f}")

    # 4. The "Preston Curve" (GDP vs Life Expectancy)
    print("\n  [3] The Preston Curve (GDP vs Life Expectancy)")
    x, y, _ = extract_pair(countries, "gdp_per_capita", "life_expectancy")
    if len(x) > 10:
        # Log fit: LE = a + b*log(GDP)
        log_x = np.log10(x)
        coeffs = np.polyfit(log_x, y, 1)
        pred = np.polyval(coeffs, log_x)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_preston = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        print(f"    LE = {coeffs[1]:.2f} + {coeffs[0]:.2f} * log10(GDP)")
        print(f"    R² = {r2_preston:.4f}  n={len(x)}")
        print(f"    => Doubling GDP adds ~{coeffs[0] * np.log10(2):.1f} years of life")
        preston_verdict = "REDISCOVERED" if r2_preston > 0.5 else "PARTIAL"
        print(f"    [{preston_verdict}]")
    else:
        r2_preston = 0
        preston_verdict = "NO_DATA"

    # 5. Fertility-income transition
    print("\n  [4] Demographic transition (GDP vs Fertility)")
    x, y, _ = extract_pair(countries, "gdp_per_capita", "fertility_rate")
    if len(x) > 10:
        fit = fit_log_log(x, y)
        if fit:
            print(f"    Fertility ~ GDP^{fit['alpha']:.4f}  R²={fit['r2']:.4f}")
            print(f"    => Richer countries have fewer children (power: {fit['alpha']:.3f})")

    # 6. Latitude and wealth
    print("\n  [5] Geography and wealth")
    x, y, _ = extract_pair(countries, "abs_latitude", "gdp_per_capita")
    if len(x) > 10:
        # Try quadratic (peak at mid-latitudes)
        coeffs = np.polyfit(x, np.log10(y), 2)
        pred = np.polyval(coeffs, x)
        ss_res = np.sum((np.log10(y) - pred) ** 2)
        ss_tot = np.sum((np.log10(y) - np.mean(np.log10(y))) ** 2)
        r2_lat = 1.0 - ss_res / ss_tot
        peak_lat = -coeffs[1] / (2 * coeffs[0])
        print(f"    log(GDP) = {coeffs[0]:.6f}*lat² + {coeffs[1]:.4f}*lat + {coeffs[2]:.2f}")
        print(f"    R² = {r2_lat:.4f}")
        if coeffs[0] < 0:
            print(f"    Peak GDP latitude: {peak_lat:.0f} degrees (inverted U)")
        else:
            print(f"    Monotonic relationship")

    # Summary
    print("\n  " + "=" * 60)
    print("  SUMMARY")
    print("  " + "=" * 60)

    high_r2 = [r for r in results if r["r2"] > 0.3]
    low_r2 = [r for r in results if r["r2"] <= 0.3]
    print(f"\n  {len(high_r2)} relationships with R² > 0.3")
    print(f"  {len(low_r2)} relationships with R² <= 0.3")
    print(f"  Best: {results[0]['test']} (R²={results[0]['r2']:.4f})")
    print(f"  Worst: {results[-1]['test']} (R²={results[-1]['r2']:.4f})")
    print(f"\n  Comparison with physics:")
    print(f"    Kepler's Law:    R² = 0.998")
    print(f"    Best social law: R² = {results[0]['r2']:.3f}")
    print(f"    Gap: {0.998 - results[0]['r2']:.3f}")
    print(f"\n  Social systems are {(0.998 - results[0]['r2'])/results[0]['r2']*100:.0f}% noisier than orbital mechanics.")

    # Artifact
    artifact = {
        "id": "E098",
        "timestamp": now,
        "world": "social_geography",
        "data_source": "World Bank Open Data API",
        "data_url": "https://data.worldbank.org/",
        "status": "passed",
        "design": {
            "description": "Search for mathematical relationships between geography, climate, economics, and human behavior across 200+ countries",
            "n_countries": n_countries,
            "n_with_latitude": n_with_lat,
            "indicators": list(INDICATORS.values()),
        },
        "result": {
            "n_tests": len(results),
            "n_significant": len(high_r2),
            "all_results": results,
            "preston_curve": {
                "r2": float(r2_preston),
                "verdict": preston_verdict,
            },
            "best_result": results[0] if results else None,
            "key_finding": f"Best social law R²={results[0]['r2']:.3f} vs Kepler R²=0.998" if results else "No results",
        },
    }

    out_path = ROOT / "results" / "E098_human_geography.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
