#!/usr/bin/env python3
"""
E101 — Zipf's Law in City Populations

Question: Does the rank-size distribution of cities follow a power law?
Is the Zipf exponent truly universal across countries?

Background:
  In 1949, George Zipf observed that if you rank cities by population,
  the nth city has population ~ P_max / n^alpha, with alpha ≈ 1.

  This means:
  - 2nd city ≈ half the population of the 1st
  - 3rd city ≈ one-third
  - 10th city ≈ one-tenth

  This is eerily precise and holds across countries, centuries, and
  cultures. Nobody has a satisfying explanation for WHY.

  Zipf's law also appears in:
  - Word frequencies in language
  - Website traffic
  - Earthquake magnitudes (Gutenberg-Richter is a Zipf law!)
  - Income distribution
  - Gene expression levels

Data: Top cities by population for multiple countries
Source: UN World Urbanization Prospects 2018 + national census data
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── City Population Data ────────────────────────────────────────
# Top cities by metropolitan area population (approximate, ~2020)
# Sources: UN, national census bureaus, citypopulation.de

COUNTRIES = {
    "USA": {
        "cities": [
            ("New York", 20140),
            ("Los Angeles", 13201),
            ("Chicago", 9459),
            ("Dallas", 7637),
            ("Houston", 7123),
            ("Washington DC", 6280),
            ("Philadelphia", 6245),
            ("Miami", 6167),
            ("Atlanta", 6020),
            ("Boston", 4875),
            ("San Francisco", 4749),
            ("Phoenix", 4946),
            ("Riverside", 4600),
            ("Detroit", 4393),
            ("Seattle", 3980),
            ("Minneapolis", 3640),
            ("San Diego", 3338),
            ("Tampa", 3176),
            ("Denver", 2963),
            ("St. Louis", 2804),
        ],
    },
    "China": {
        "cities": [
            ("Shanghai", 28516),
            ("Beijing", 21542),
            ("Chongqing", 17384),
            ("Guangzhou", 16096),
            ("Shenzhen", 14678),
            ("Tianjin", 13866),
            ("Chengdu", 12982),
            ("Wuhan", 12327),
            ("Hangzhou", 11067),
            ("Dongguan", 10465),
            ("Nanjing", 9425),
            ("Suzhou", 8847),
            ("Shenyang", 8294),
            ("Xi'an", 8000),
            ("Harbin", 7964),
            ("Qingdao", 7172),
            ("Dalian", 6750),
            ("Zhengzhou", 6640),
            ("Jinan", 6320),
            ("Changsha", 6018),
        ],
    },
    "Brazil": {
        "cities": [
            ("Sao Paulo", 22043),
            ("Rio de Janeiro", 13458),
            ("Belo Horizonte", 6000),
            ("Brasilia", 4804),
            ("Recife", 4079),
            ("Porto Alegre", 4106),
            ("Fortaleza", 4074),
            ("Salvador", 3953),
            ("Curitiba", 3573),
            ("Campinas", 3243),
            ("Goiania", 2570),
            ("Manaus", 2220),
            ("Belem", 2478),
            ("Vitoria", 2052),
            ("Santos", 1853),
        ],
    },
    "India": {
        "cities": [
            ("Mumbai", 20668),
            ("Delhi", 16787),
            ("Bangalore", 12327),
            ("Hyderabad", 10269),
            ("Ahmedabad", 8009),
            ("Chennai", 7088),
            ("Kolkata", 14974),
            ("Surat", 6538),
            ("Pune", 6629),
            ("Jaipur", 3971),
            ("Lucknow", 3600),
            ("Kanpur", 3124),
            ("Nagpur", 2893),
            ("Indore", 2468),
            ("Patna", 2321),
        ],
    },
    "Japan": {
        "cities": [
            ("Tokyo", 37393),
            ("Osaka", 19281),
            ("Nagoya", 9507),
            ("Fukuoka", 5539),
            ("Sapporo", 2670),
            ("Sendai", 2321),
            ("Hiroshima", 2067),
            ("Kitakyushu", 1814),
            ("Kumamoto", 1492),
            ("Niigata", 1395),
            ("Hamamatsu", 1300),
            ("Okayama", 1233),
            ("Shizuoka", 1200),
            ("Kagoshima", 1100),
            ("Kanazawa", 1050),
        ],
    },
    "Germany": {
        "cities": [
            ("Berlin", 3645),
            ("Hamburg", 1899),
            ("Munich", 1472),
            ("Cologne", 1086),
            ("Frankfurt", 753),
            ("Stuttgart", 635),
            ("Dusseldorf", 619),
            ("Leipzig", 593),
            ("Dortmund", 587),
            ("Essen", 583),
            ("Bremen", 567),
            ("Dresden", 556),
            ("Hannover", 538),
            ("Nuremberg", 518),
            ("Duisburg", 498),
        ],
    },
    "Mexico": {
        "cities": [
            ("Mexico City", 21782),
            ("Guadalajara", 5260),
            ("Monterrey", 5085),
            ("Puebla", 3199),
            ("Toluca", 2353),
            ("Tijuana", 2010),
            ("Leon", 1847),
            ("Ciudad Juarez", 1512),
            ("Torreon", 1408),
            ("Queretaro", 1323),
            ("San Luis Potosi", 1222),
            ("Merida", 1142),
            ("Aguascalientes", 1065),
            ("Tampico", 918),
            ("Chihuahua", 878),
        ],
    },
    "France": {
        "cities": [
            ("Paris", 11020),
            ("Lyon", 1719),
            ("Marseille", 1608),
            ("Toulouse", 1044),
            ("Bordeaux", 986),
            ("Lille", 955),
            ("Nice", 610),
            ("Nantes", 607),
            ("Strasbourg", 540),
            ("Rennes", 480),
            ("Grenoble", 450),
            ("Montpellier", 440),
            ("Rouen", 430),
            ("Toulon", 412),
            ("Saint-Etienne", 395),
        ],
    },
}


def fit_zipf(populations, country_name):
    """Fit Zipf's law: P(r) = C * r^(-alpha)."""
    pop = np.array(sorted(populations, reverse=True), dtype=float)
    ranks = np.arange(1, len(pop) + 1, dtype=float)

    log_r = np.log10(ranks)
    log_p = np.log10(pop)

    coeffs = np.polyfit(log_r, log_p, 1)
    alpha = -coeffs[0]
    C = 10 ** coeffs[1]

    pred = np.polyval(coeffs, log_r)
    ss_res = np.sum((log_p - pred) ** 2)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Check rank × pop product (should be ~constant for alpha=1)
    product = ranks * pop
    product_cv = np.std(product) / np.mean(product)

    return {
        "country": country_name,
        "alpha": float(alpha),
        "C": float(C),
        "r2": float(r2),
        "n_cities": len(pop),
        "largest": float(pop[0]),
        "smallest": float(pop[-1]),
        "ratio_1_2": float(pop[0] / pop[1]),
        "rank_pop_product_mean": float(np.mean(product)),
        "rank_pop_product_cv": float(product_cv),
    }


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E101 -- Zipf's Law in City Populations")
    print("=" * 70)

    n_countries = len(COUNTRIES)
    total_cities = sum(len(c["cities"]) for c in COUNTRIES.values())
    print(f"\n  Data: {total_cities} cities across {n_countries} countries")

    # Fit each country
    results = []
    print(f"\n  {'Country':12s} {'alpha':>7s} {'R2':>8s} {'Cities':>7s} {'Ratio 1/2':>10s} {'Largest':>15s}")
    print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*7} {'-'*10} {'-'*15}")

    for name, data in COUNTRIES.items():
        pops = [c[1] for c in data["cities"]]
        fit = fit_zipf(pops, name)
        results.append(fit)

        city1 = data["cities"][0][0]
        print(f"  {name:12s} {fit['alpha']:7.4f} {fit['r2']:8.4f} {fit['n_cities']:7d} {fit['ratio_1_2']:10.2f} {city1}")

    # Summary statistics
    alphas = [r["alpha"] for r in results]
    r2s = [r["r2"] for r in results]

    print(f"\n  " + "=" * 50)
    print(f"  ZIPF EXPONENT ANALYSIS")
    print(f"  " + "=" * 50)

    print(f"\n  Mean alpha:   {np.mean(alphas):.4f}  (Zipf predicts: 1.0)")
    print(f"  Std alpha:    {np.std(alphas):.4f}")
    print(f"  Range:        {min(alphas):.4f} to {max(alphas):.4f}")
    print(f"  Mean R2:      {np.mean(r2s):.4f}")

    # Is alpha ≈ 1?
    mean_err = abs(np.mean(alphas) - 1.0)
    zipf_verdict = "REDISCOVERED" if mean_err < 0.2 and np.mean(r2s) > 0.9 else "PARTIAL"
    print(f"\n  Distance from pure Zipf (alpha=1): {mean_err:.4f}")
    print(f"  [{zipf_verdict}]")

    # Primacy analysis (is the largest city "too big"?)
    print(f"\n  " + "=" * 50)
    print(f"  PRIMACY ANALYSIS")
    print(f"  " + "=" * 50)
    print(f"\n  If Zipf holds perfectly, city #1 should be 2x city #2.")
    print(f"  Actual ratios:")
    for r in sorted(results, key=lambda x: -x["ratio_1_2"]):
        label = ""
        if r["ratio_1_2"] > 3:
            label = " <-- PRIMATE CITY"
        elif r["ratio_1_2"] < 1.3:
            label = " <-- balanced"
        print(f"    {r['country']:12s}: {r['ratio_1_2']:.2f}x{label}")

    primate = [r for r in results if r["ratio_1_2"] > 3]
    balanced = [r for r in results if r["ratio_1_2"] < 2]
    print(f"\n  Primate cities (ratio > 3x): {len(primate)} countries")
    print(f"  Balanced (ratio < 2x): {len(balanced)} countries")

    # Global fit (all cities combined)
    print(f"\n  " + "=" * 50)
    print(f"  GLOBAL FIT (all {total_cities} cities)")
    print(f"  " + "=" * 50)
    all_pops = []
    for data in COUNTRIES.values():
        all_pops.extend([c[1] for c in data["cities"]])

    global_fit = fit_zipf(all_pops, "GLOBAL")
    print(f"  alpha = {global_fit['alpha']:.4f}")
    print(f"  R2 = {global_fit['r2']:.4f}")

    # Connection to other Zipf phenomena
    print(f"\n  " + "=" * 50)
    print(f"  ZIPF'S LAW APPEARS EVERYWHERE")
    print(f"  " + "=" * 50)
    print(f"  Cities (this experiment):     alpha ~ {np.mean(alphas):.2f}")
    print(f"  Word frequencies:             alpha ~ 1.00")
    print(f"  Website traffic:              alpha ~ 1.00")
    print(f"  Earthquake magnitudes (E092): b ~ 0.81 (related)")
    print(f"  Income distribution:          alpha ~ 1.5-2.0 (Pareto)")
    print(f"  Gene expression:              alpha ~ 1.0-1.5")
    print(f"\n  Why? Nobody knows. It emerges from ANY process where")
    print(f"  growth is proportional to current size (preferential attachment).")

    # Artifact
    artifact = {
        "id": "E101",
        "timestamp": now,
        "world": "social",
        "data_source": "UN World Urbanization Prospects + national census",
        "data_url": "https://population.un.org/wup/",
        "status": "passed" if zipf_verdict == "REDISCOVERED" else "partial",
        "design": {
            "description": "Test Zipf's law on city population distributions across 8 countries",
            "n_countries": n_countries,
            "n_cities": total_cities,
        },
        "result": {
            "country_fits": results,
            "mean_alpha": float(np.mean(alphas)),
            "std_alpha": float(np.std(alphas)),
            "mean_r2": float(np.mean(r2s)),
            "global_fit": global_fit,
            "verdict": zipf_verdict,
            "key_finding": f"Zipf alpha = {np.mean(alphas):.3f} +/- {np.std(alphas):.3f} across {n_countries} countries",
        },
    }

    out_path = ROOT / "results" / "E101_zipf_cities.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
