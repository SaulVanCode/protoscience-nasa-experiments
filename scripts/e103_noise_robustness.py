#!/usr/bin/env python3
"""
E103 — Noise Robustness: At What Point Does Discovery Break?

Question: How much noise can ProtoScience tolerate before it starts
proposing garbage? Where is the "death line" for each law?

Design:
  Take 5 known laws we've already rediscovered.
  Add Gaussian noise at 0%, 1%, 2%, 5%, 10%, 20%, 50%, 100%.
  For each noise level, re-fit and measure R².
  Find the exact noise threshold where R² drops below 0.9.

This is the "stress test" that reviewers will ask for.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00]
N_TRIALS = 20  # repeat each noise level to get statistics


def add_noise(y, noise_frac, rng):
    """Add Gaussian noise as a fraction of each value."""
    noise = rng.normal(0, 1, len(y)) * noise_frac * np.abs(y)
    return y + noise


def fit_power_law(x, y):
    """Fit y = C * x^alpha, return R²."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return 0.0, 0.0, 0.0
    lx, ly = np.log10(x[mask]), np.log10(y[mask])
    coeffs = np.polyfit(lx, ly, 1)
    alpha = coeffs[0]
    C = 10 ** coeffs[1]
    pred = np.polyval(coeffs, lx)
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r2), float(alpha), float(C)


def fit_exponential(x, y):
    """Fit y = C * exp(beta * x), return R²."""
    mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return 0.0, 0.0, 0.0
    log_y = np.log(y[mask])
    coeffs = np.polyfit(x[mask], log_y, 1)
    beta = coeffs[0]
    C = np.exp(coeffs[1])
    pred = np.polyval(coeffs, x[mask])
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r2), float(beta), float(C)


def fit_linear(x, y):
    """Linear fit y = a + b*x."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return 0.0, 0.0, 0.0
    coeffs = np.polyfit(x[mask], y[mask], 1)
    pred = np.polyval(coeffs, x[mask])
    ss_res = np.sum((y[mask] - pred) ** 2)
    ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(r2), float(coeffs[0]), float(coeffs[1])


# ── Test Laws ───────────────────────────────────────────────────

def generate_kepler(n=200):
    """Kepler's Third Law: P² = a³/M★"""
    rng = np.random.RandomState(42)
    a = rng.uniform(0.1, 50, n)  # semi-major axis (AU)
    M = rng.uniform(0.3, 3.0, n)  # stellar mass (solar)
    P = np.sqrt(a ** 3 / M)  # period (years)
    return a ** 3 / M, P ** 2, "power_law", 1.0, "Kepler P²=a³/M★"


def generate_stefan_boltzmann(n=200):
    """Stefan-Boltzmann: L ~ T⁴"""
    rng = np.random.RandomState(43)
    T = rng.uniform(3000, 30000, n)  # temperature (K)
    L = 5.67e-8 * T ** 4  # luminosity (arbitrary units)
    return T, L, "power_law", 4.0, "Stefan-Boltzmann L~T⁴"


def generate_gutenberg_richter(n=500):
    """Gutenberg-Richter: log N = a - bM (b≈1)"""
    rng = np.random.RandomState(44)
    # Generate magnitudes following GR distribution
    b = 1.0
    mags = -np.log10(rng.uniform(0, 1, n)) / b + 2.0
    mags = mags[mags < 8.0]
    mag_bins = np.arange(2.0, 7.5, 0.1)
    cumulative = np.array([np.sum(mags >= m) for m in mag_bins])
    mask = cumulative > 0
    return mag_bins[mask], cumulative[mask].astype(float), "power_law", -1.0, "Gutenberg-Richter log(N)=a-bM"


def generate_gompertz(n=50):
    """Gompertz: m(x) = alpha * exp(beta * x)"""
    ages = np.linspace(30, 95, n)
    alpha = 0.00008
    beta = 0.085
    mx = alpha * np.exp(beta * ages)
    return ages, mx, "exponential", beta, "Gompertz mortality"


def generate_hubble(n=150):
    """Hubble's Law: v = H0 * d"""
    rng = np.random.RandomState(46)
    d = rng.uniform(5, 300, n)  # Mpc
    H0 = 70.0
    v = H0 * d  # km/s
    return d, v, "linear", H0, "Hubble v=H₀d"


LAWS = [
    generate_kepler,
    generate_stefan_boltzmann,
    generate_gutenberg_richter,
    generate_gompertz,
    generate_hubble,
]


def stress_test_law(gen_func):
    """Run noise stress test on a single law."""
    x_clean, y_clean, fit_type, expected_param, name = gen_func()

    results = []
    for noise in NOISE_LEVELS:
        r2_trials = []
        param_trials = []

        for trial in range(N_TRIALS):
            rng = np.random.RandomState(trial * 1000 + int(noise * 100))
            y_noisy = add_noise(y_clean.copy(), noise, rng)

            if fit_type == "power_law":
                r2, param, _ = fit_power_law(x_clean, y_noisy)
            elif fit_type == "exponential":
                r2, param, _ = fit_exponential(x_clean, y_noisy)
            elif fit_type == "linear":
                r2, param, _ = fit_linear(x_clean, y_noisy)
            else:
                r2, param = 0, 0

            r2_trials.append(r2)
            param_trials.append(param)

        mean_r2 = float(np.mean(r2_trials))
        std_r2 = float(np.std(r2_trials))
        mean_param = float(np.mean(param_trials))
        param_err = abs(mean_param - expected_param) / abs(expected_param) * 100 if expected_param != 0 else 0

        results.append({
            "noise_pct": float(noise * 100),
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "mean_param": mean_param,
            "param_error_pct": float(param_err),
        })

    # Find death line (where R² first drops below 0.9)
    death_noise = None
    for r in results:
        if r["mean_r2"] < 0.9:
            death_noise = r["noise_pct"]
            break

    return {
        "law": name,
        "fit_type": fit_type,
        "expected_param": float(expected_param),
        "n_points": len(x_clean),
        "noise_curve": results,
        "death_line_pct": death_noise,
    }


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E103 -- Noise Robustness: At What Point Does Discovery Break?")
    print("=" * 70)

    all_results = []

    for gen_func in LAWS:
        result = stress_test_law(gen_func)
        all_results.append(result)

        print(f"\n  {result['law']} (n={result['n_points']}, {result['fit_type']})")
        print(f"  {'Noise':>7s} {'R²':>8s} {'±':>6s} {'Param err':>10s}")
        print(f"  {'-'*7} {'-'*8} {'-'*6} {'-'*10}")

        for r in result["noise_curve"]:
            bar = "#" * int(r["mean_r2"] * 30)
            dead = " << DEAD" if r["mean_r2"] < 0.9 else ""
            print(f"  {r['noise_pct']:6.0f}% {r['mean_r2']:8.4f} {r['std_r2']:6.4f} {r['param_error_pct']:9.2f}%  |{bar}{dead}")

        if result["death_line_pct"] is not None:
            print(f"  --> Death line: {result['death_line_pct']:.0f}% noise")
        else:
            print(f"  --> Survives all noise levels up to {NOISE_LEVELS[-1]*100:.0f}%")

    # Summary
    print("\n  " + "=" * 60)
    print("  NOISE ROBUSTNESS RANKING")
    print("  " + "=" * 60)

    ranked = sorted(all_results, key=lambda r: -(r["death_line_pct"] or 999))
    print(f"\n  {'Law':35s} {'Death line':>12s} {'Type':>12s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    for r in ranked:
        dl = f"{r['death_line_pct']:.0f}%" if r["death_line_pct"] else "IMMORTAL"
        print(f"  {r['law']:35s} {dl:>12s} {r['fit_type']:>12s}")

    # Key insight
    print(f"\n  Key finding:")
    most_robust = ranked[0]
    least_robust = ranked[-1]
    print(f"    Most robust:  {most_robust['law']} (dies at {most_robust['death_line_pct'] or '>200'}% noise)")
    print(f"    Least robust: {least_robust['law']} (dies at {least_robust['death_line_pct'] or '>200'}% noise)")

    # At 10% noise, which laws survive?
    print(f"\n  At 10% noise:")
    for r in all_results:
        entry = [e for e in r["noise_curve"] if e["noise_pct"] == 10.0]
        if entry:
            e = entry[0]
            status = "ALIVE" if e["mean_r2"] >= 0.9 else "DEAD"
            print(f"    {r['law']:35s}  R²={e['mean_r2']:.4f}  param_err={e['param_error_pct']:.1f}%  [{status}]")

    # Artifact
    artifact = {
        "id": "E103",
        "timestamp": now,
        "world": "methodology",
        "data_source": "Synthetic (known laws + controlled noise)",
        "status": "passed",
        "design": {
            "description": "Stress test: add 0-200% Gaussian noise to 5 known laws, measure R² degradation and parameter drift",
            "noise_levels": [n * 100 for n in NOISE_LEVELS],
            "n_trials_per_level": N_TRIALS,
            "laws_tested": [r["law"] for r in all_results],
        },
        "result": {
            "law_results": all_results,
            "ranking": [{"law": r["law"], "death_line": r["death_line_pct"]} for r in ranked],
        },
    }

    out_path = ROOT / "results" / "E103_noise_robustness.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
