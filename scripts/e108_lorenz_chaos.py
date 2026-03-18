#!/usr/bin/env python3
"""
E108 — Finding Order in Chaos: The Lorenz Attractor

Question: Can ProtoScience recover the exact differential equations
of a chaotic system from a noisy trajectory?

Background:
  In 1963, Edward Lorenz discovered that a simplified model of
  atmospheric convection produces chaotic behavior — tiny changes
  in initial conditions lead to wildly different outcomes.

  The Lorenz system:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

  With sigma=10, rho=28, beta=8/3, the system is chaotic.

  The trajectory LOOKS random, but it's governed by 3 simple
  equations with only 7 terms total. The "butterfly" attractor
  is deterministic — not random.

  This is the ultimate test: can sparse regression find the
  EXACT equations hiding inside apparent chaos?

Source: Lorenz (1963), "Deterministic Nonperiodic Flow"
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent

# True Lorenz parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz(t, state):
    """Lorenz system ODE."""
    x, y, z = state
    return [
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ]


def generate_trajectory(x0, t_span, dt, noise_frac=0.0):
    """Generate a Lorenz trajectory."""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(lorenz, t_span, x0, t_eval=t_eval, method="RK45",
                    rtol=1e-10, atol=1e-12)

    x, y, z = sol.y
    t = sol.t

    # Add noise
    if noise_frac > 0:
        rng = np.random.RandomState(42)
        x += rng.normal(0, noise_frac * np.std(x), len(x))
        y += rng.normal(0, noise_frac * np.std(y), len(y))
        z += rng.normal(0, noise_frac * np.std(z), len(z))

    return t, x, y, z


def estimate_derivatives(t, x, y, z):
    """Estimate dx/dt, dy/dt, dz/dt using central differences."""
    dt = t[1] - t[0]
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    dz = np.gradient(z, dt)
    return dx, dy, dz


def sindy_fit(t, x, y, z, dx, dy, dz, threshold=0.5):
    """
    Sparse regression (SINDy-like) to find governing equations.

    Library: 1, x, y, z, x², y², z², xy, xz, yz
    """
    n = len(x)

    # Build library matrix
    lib_names = ["1", "x", "y", "z", "x²", "y²", "z²", "xy", "xz", "yz"]
    Theta = np.column_stack([
        np.ones(n),    # 1
        x,             # x
        y,             # y
        z,             # z
        x ** 2,        # x²
        y ** 2,        # y²
        z ** 2,        # z²
        x * y,         # xy
        x * z,         # xz
        y * z,         # yz
    ])

    results = {}

    for var_name, target in [("dx/dt", dx), ("dy/dt", dy), ("dz/dt", dz)]:
        # Sequential Thresholded Least Squares (STLS)
        Xi = np.linalg.lstsq(Theta, target, rcond=None)[0]

        # Threshold small coefficients (sparsify)
        for iteration in range(10):
            small = np.abs(Xi) < threshold
            Xi[small] = 0
            big = ~small
            if big.sum() == 0:
                break
            Xi[big] = np.linalg.lstsq(Theta[:, big], target, rcond=None)[0]

        # Compute R²
        pred = Theta @ Xi
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Build equation string
        terms = []
        for coeff, name in zip(Xi, lib_names):
            if abs(coeff) > 1e-6:
                terms.append(f"{coeff:+.4f}*{name}")

        equation = " ".join(terms) if terms else "0"

        results[var_name] = {
            "coefficients": {name: float(c) for name, c in zip(lib_names, Xi)},
            "equation": equation,
            "r2": float(r2),
            "n_terms": int(np.sum(np.abs(Xi) > 1e-6)),
        }

    return results


def verify_coefficients(sindy_results):
    """Check if SINDy found the true Lorenz parameters."""
    checks = {}

    # dx/dt = sigma*(y - x) = -sigma*x + sigma*y
    dx_coeffs = sindy_results["dx/dt"]["coefficients"]
    found_sigma_x = -dx_coeffs.get("x", 0)
    found_sigma_y = dx_coeffs.get("y", 0)
    sigma_err = (abs(found_sigma_x - SIGMA) + abs(found_sigma_y - SIGMA)) / (2 * SIGMA) * 100
    checks["sigma"] = {
        "true": SIGMA,
        "found_from_x": float(found_sigma_x),
        "found_from_y": float(found_sigma_y),
        "error_pct": float(sigma_err),
        "match": sigma_err < 5,
    }

    # dy/dt = x*(rho - z) - y = rho*x - y - xz
    dy_coeffs = sindy_results["dy/dt"]["coefficients"]
    found_rho = dy_coeffs.get("x", 0)
    found_minus1 = -dy_coeffs.get("y", 0)
    found_xz = -dy_coeffs.get("xz", 0)
    rho_err = abs(found_rho - RHO) / RHO * 100
    checks["rho"] = {
        "true": RHO,
        "found": float(found_rho),
        "error_pct": float(rho_err),
        "match": rho_err < 5,
    }
    checks["dy_y_coeff"] = {
        "true": -1.0,
        "found": float(dy_coeffs.get("y", 0)),
        "match": abs(dy_coeffs.get("y", 0) + 1.0) < 0.2,
    }

    # dz/dt = xy - beta*z
    dz_coeffs = sindy_results["dz/dt"]["coefficients"]
    found_beta = -dz_coeffs.get("z", 0)
    found_xy = dz_coeffs.get("xy", 0)
    beta_err = abs(found_beta - BETA) / BETA * 100
    checks["beta"] = {
        "true": float(BETA),
        "found": float(found_beta),
        "error_pct": float(beta_err),
        "match": beta_err < 5,
    }
    checks["dz_xy_coeff"] = {
        "true": 1.0,
        "found": float(found_xy),
        "match": abs(found_xy - 1.0) < 0.2,
    }

    return checks


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E108 -- Finding Order in Chaos: The Lorenz Attractor")
    print("=" * 70)

    # 1. Generate clean trajectory
    print(f"\n  [1] Generating Lorenz trajectory...")
    print(f"    True parameters: sigma={SIGMA}, rho={RHO}, beta={BETA:.4f}")
    x0 = [-8.0, 8.0, 27.0]
    t, x, y, z = generate_trajectory(x0, (0, 50), dt=0.001)
    print(f"    {len(t)} time steps, dt=0.001")
    print(f"    x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"    y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"    z range: [{z.min():.2f}, {z.max():.2f}]")

    # 2. Estimate derivatives
    print(f"\n  [2] Estimating derivatives...")
    dx, dy, dz = estimate_derivatives(t, x, y, z)

    # 3. SINDy (clean data)
    print(f"\n  [3] Sparse regression (SINDy) on CLEAN data:")
    results_clean = sindy_fit(t, x, y, z, dx, dy, dz, threshold=0.5)

    for var, res in results_clean.items():
        print(f"\n    {var} = {res['equation']}")
        print(f"    R² = {res['r2']:.8f}  ({res['n_terms']} terms)")

    # 4. Verify coefficients
    print(f"\n  [4] Coefficient verification (clean):")
    checks_clean = verify_coefficients(results_clean)

    n_matched = 0
    for name, check in checks_clean.items():
        if "match" in check:
            status = "OK" if check["match"] else "MISS"
            if check["match"]:
                n_matched += 1
            found_val = check.get("found", check.get("found_from_x", 0))
            if "error_pct" in check:
                print(f"    [{status}] {name:15s}: true={check['true']:.4f}, found={found_val:.4f}, err={check['error_pct']:.2f}%")
            else:
                print(f"    [{status}] {name:15s}: true={check['true']:.4f}, found={found_val:.4f}")

    # 5. Now with noise
    noise_results = {}
    for noise_pct in [0.01, 0.05, 0.10, 0.20]:
        t_n, x_n, y_n, z_n = generate_trajectory(x0, (0, 50), dt=0.001, noise_frac=noise_pct)
        dx_n, dy_n, dz_n = estimate_derivatives(t_n, x_n, y_n, z_n)
        res_n = sindy_fit(t_n, x_n, y_n, z_n, dx_n, dy_n, dz_n, threshold=0.5)
        checks_n = verify_coefficients(res_n)

        mean_r2 = np.mean([res_n[v]["r2"] for v in res_n])
        n_match = sum(1 for c in checks_n.values() if c.get("match", False))

        noise_results[f"{noise_pct*100:.0f}%"] = {
            "mean_r2": float(mean_r2),
            "n_params_matched": n_match,
            "total_params": len(checks_n),
        }

    print(f"\n  [5] Noise robustness:")
    print(f"    {'Noise':>7s} {'Mean R²':>9s} {'Params OK':>10s}")
    print(f"    {'-'*7} {'-'*9} {'-'*10}")
    for noise, nr in noise_results.items():
        print(f"    {noise:>7s} {nr['mean_r2']:9.6f} {nr['n_params_matched']}/{nr['total_params']}")

    # 6. The butterfly effect
    print(f"\n  [6] The Butterfly Effect:")
    t1, x1, _, _ = generate_trajectory([-8.0, 8.0, 27.0], (0, 30), dt=0.001)
    t2, x2, _, _ = generate_trajectory([-8.0, 8.000001, 27.0], (0, 30), dt=0.001)  # epsilon change
    n_min = min(len(x1), len(x2))

    divergence_time = None
    for i in range(n_min):
        if abs(x1[i] - x2[i]) > 5.0:
            divergence_time = t1[i]
            break

    if divergence_time:
        print(f"    Initial difference: 0.000001 (one millionth)")
        print(f"    Trajectories diverge after {divergence_time:.1f} time units")
        print(f"    After divergence, prediction is impossible")
        print(f"    BUT the underlying equations are still perfectly deterministic")

    # 7. True vs expected equations
    print(f"\n  [7] The Lorenz equations (as discovered by SINDy):")
    print(f"\n    True:      dx/dt = {SIGMA:.1f}*(y - x)")
    print(f"    Found:     {results_clean['dx/dt']['equation']}")
    print(f"\n    True:      dy/dt = x*({RHO:.1f} - z) - y")
    print(f"    Found:     {results_clean['dy/dt']['equation']}")
    print(f"\n    True:      dz/dt = x*y - {BETA:.4f}*z")
    print(f"    Found:     {results_clean['dz/dt']['equation']}")

    # Summary
    total_checks = len(checks_clean)
    print(f"\n  " + "=" * 60)
    print(f"  SUMMARY")
    print(f"  " + "=" * 60)
    verdict = "REDISCOVERED" if n_matched >= 4 else "PARTIAL"
    print(f"\n  Lorenz system: {n_matched}/{total_checks} parameters recovered [{verdict}]")
    print(f"  Mean R² (clean): {np.mean([r['r2'] for r in results_clean.values()]):.8f}")
    print(f"  The 'skeleton' of chaos: 3 equations, 7 terms, deterministic")
    print(f"  Chaos is not randomness — it's simple rules with extreme sensitivity")

    # Artifact
    artifact = {
        "id": "E108",
        "timestamp": now,
        "world": "chaos",
        "data_source": "Simulated Lorenz system (Lorenz 1963)",
        "status": "passed" if verdict == "REDISCOVERED" else "partial",
        "design": {
            "description": "Recover the Lorenz differential equations from a chaotic trajectory using sparse regression (SINDy)",
            "true_params": {"sigma": SIGMA, "rho": RHO, "beta": float(BETA)},
            "n_timesteps": len(t),
        },
        "result": {
            "clean_fit": {k: {"equation": v["equation"], "r2": v["r2"], "n_terms": v["n_terms"]}
                         for k, v in results_clean.items()},
            "coefficient_checks": {k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else vv) for kk, vv in v.items()}
                                   for k, v in checks_clean.items()},
            "noise_robustness": noise_results,
            "butterfly_divergence_time": float(divergence_time) if divergence_time else None,
            "verdict": verdict,
            "n_matched": n_matched,
        },
    }

    out_path = ROOT / "results" / "E108_lorenz_chaos.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
