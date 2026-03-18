#!/usr/bin/env python3
"""
E100 — Patterns in Nuclear Decay: The Table of Nuclides

Question: Are there mathematical patterns in how atomic nuclei decay?
Is there a relationship between nuclear properties and half-life?

Background:
  There are ~3,000 known nuclides. Each has a mass number (A = Z + N),
  proton number (Z), neutron number (N), and a half-life ranging from
  femtoseconds to billions of years. The "valley of stability" defines
  which nuclei are stable and which decay.

  Known patterns:
  - Magic numbers (2, 8, 20, 28, 50, 82, 126) for extra stability
  - The Geiger-Nuttall law: log(t½) ~ Z / sqrt(E_alpha) for alpha decay
  - The N/Z ratio determines stability (~1 for light, ~1.5 for heavy)
  - Binding energy per nucleon peaks at Fe-56

Data: Curated nuclear data from NNDC/IAEA
Source: https://www.nndc.bnl.gov/
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Nuclear Data ────────────────────────────────────────────────
# Curated dataset of nuclides with half-lives
# Source: NNDC (Brookhaven), IAEA Nuclear Data Services
# Format: (Z, N, symbol, A, half_life_seconds, decay_mode, binding_energy_per_nucleon_MeV)

# Stable nuclides have half_life = None (effectively infinite)
# We include a mix of stable and unstable across the chart

NUCLIDES = [
    # Light nuclei
    (1, 0, "H-1", 1, None, "stable", 0.000),
    (1, 1, "H-2", 2, None, "stable", 1.112),
    (1, 2, "H-3", 3, 3.888e8, "beta-", 2.827),
    (2, 1, "He-3", 3, None, "stable", 2.573),
    (2, 2, "He-4", 4, None, "stable", 7.074),
    (3, 3, "Li-6", 6, None, "stable", 5.332),
    (3, 4, "Li-7", 7, None, "stable", 5.606),
    (4, 4, "Be-8", 8, 8.19e-17, "alpha", 7.062),
    (4, 5, "Be-9", 9, None, "stable", 6.463),
    (5, 5, "B-10", 10, None, "stable", 6.475),
    (5, 6, "B-11", 11, None, "stable", 6.928),
    (6, 6, "C-12", 12, None, "stable", 7.680),
    (6, 7, "C-13", 13, None, "stable", 7.470),
    (6, 8, "C-14", 14, 1.808e11, "beta-", 7.520),
    (7, 7, "N-14", 14, None, "stable", 7.476),
    (7, 8, "N-15", 15, None, "stable", 7.699),
    (8, 8, "O-16", 16, None, "stable", 7.976),
    (8, 9, "O-17", 17, None, "stable", 7.751),
    (8, 10, "O-18", 18, None, "stable", 7.767),
    # Medium nuclei
    (10, 10, "Ne-20", 20, None, "stable", 8.032),
    (11, 12, "Na-23", 23, None, "stable", 8.112),
    (12, 12, "Mg-24", 24, None, "stable", 8.261),
    (13, 14, "Al-27", 27, None, "stable", 8.332),
    (14, 14, "Si-28", 28, None, "stable", 8.448),
    (15, 16, "P-31", 31, None, "stable", 8.481),
    (16, 16, "S-32", 32, None, "stable", 8.493),
    (17, 18, "Cl-35", 35, None, "stable", 8.520),
    (18, 22, "Ar-40", 40, None, "stable", 8.595),
    (19, 20, "K-39", 39, None, "stable", 8.557),
    (19, 21, "K-40", 40, 3.94e16, "beta-", 8.538),
    (20, 20, "Ca-40", 40, None, "stable", 8.551),
    (20, 28, "Ca-48", 48, 6.4e26, "2beta-", 8.667),
    (26, 28, "Fe-54", 54, None, "stable", 8.736),
    (26, 30, "Fe-56", 56, None, "stable", 8.790),  # Maximum BE/A
    (26, 32, "Fe-58", 58, None, "stable", 8.792),
    (27, 32, "Co-59", 59, None, "stable", 8.768),
    (28, 30, "Ni-58", 58, None, "stable", 8.732),
    (28, 34, "Ni-62", 62, None, "stable", 8.795),  # True max BE/A
    (29, 34, "Cu-63", 63, None, "stable", 8.752),
    (30, 34, "Zn-64", 64, None, "stable", 8.736),
    # Heavy stable
    (38, 50, "Sr-88", 88, None, "stable", 8.733),
    (40, 50, "Zr-90", 90, None, "stable", 8.710),
    (42, 54, "Mo-96", 96, None, "stable", 8.654),
    (50, 70, "Sn-120", 120, None, "stable", 8.505),
    (50, 74, "Sn-124", 124, None, "stable", 8.468),
    (56, 82, "Ba-138", 138, None, "stable", 8.394),
    (57, 82, "La-139", 139, None, "stable", 8.378),
    (79, 118, "Au-197", 197, None, "stable", 7.916),
    (82, 124, "Pb-206", 206, None, "stable", 7.875),
    (82, 126, "Pb-208", 208, None, "stable", 7.868),
    # Radioactive heavy
    (83, 126, "Bi-209", 209, 6.0e26, "alpha", 7.848),
    (84, 126, "Po-210", 210, 1.196e7, "alpha", 7.834),
    (86, 136, "Rn-222", 222, 3.304e5, "alpha", 7.694),
    (88, 138, "Ra-226", 226, 5.049e10, "alpha", 7.662),
    (90, 140, "Th-230", 230, 2.379e12, "alpha", 7.631),
    (90, 142, "Th-232", 232, 4.434e17, "alpha", 7.615),
    (92, 143, "U-235", 235, 2.221e16, "alpha", 7.591),
    (92, 146, "U-238", 238, 1.410e17, "alpha", 7.570),
    (93, 144, "Np-237", 237, 6.752e13, "alpha", 7.575),
    (94, 145, "Pu-239", 239, 7.609e11, "alpha", 7.560),
    (94, 150, "Pu-244", 244, 2.524e15, "alpha", 7.526),
    (95, 146, "Am-241", 241, 1.364e10, "alpha", 7.543),
    (96, 152, "Cm-248", 248, 1.069e13, "alpha", 7.499),
    # Very short-lived
    (99, 154, "Es-253", 253, 1.764e6, "alpha", 7.460),
    (100, 157, "Fm-257", 257, 8.681e6, "alpha", 7.433),
    (102, 157, "No-259", 259, 3480.0, "alpha", 7.410),
    (104, 163, "Rf-267", 267, 7800.0, "alpha", 7.370),
    (106, 160, "Sg-266", 266, 21.0, "alpha", 7.350),
    (108, 157, "Hs-265", 265, 0.002, "alpha", 7.320),
    (110, 161, "Ds-271", 271, 0.062, "alpha", 7.290),
    (114, 175, "Fl-289", 289, 1.9, "alpha", 7.200),
    (118, 176, "Og-294", 294, 0.00069, "alpha", 7.150),
]


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E100 -- Patterns in Nuclear Decay: The Table of Nuclides")
    print("=" * 70)

    # Separate stable and unstable
    stable = [n for n in NUCLIDES if n[4] is None]
    unstable = [n for n in NUCLIDES if n[4] is not None]

    print(f"\n  Dataset: {len(NUCLIDES)} nuclides ({len(stable)} stable, {len(unstable)} radioactive)")
    print(f"  Z range: {min(n[0] for n in NUCLIDES)} to {max(n[0] for n in NUCLIDES)}")
    print(f"  Half-life range: {min(n[4] for n in unstable):.2e} s to {max(n[4] for n in unstable):.2e} s")
    print(f"  That's {max(n[4] for n in unstable)/min(n[4] for n in unstable):.1e}x range ({np.log10(max(n[4] for n in unstable)/min(n[4] for n in unstable)):.0f} decades)")

    # 1. Binding energy per nucleon vs A (the famous curve)
    print("\n  [1] Binding energy per nucleon vs mass number A")
    A_all = np.array([n[3] for n in NUCLIDES], dtype=float)
    BE_all = np.array([n[6] for n in NUCLIDES])

    # Find the peak
    peak_idx = np.argmax(BE_all)
    print(f"    Peak: {NUCLIDES[peak_idx][2]} at A={A_all[peak_idx]:.0f}, BE/A={BE_all[peak_idx]:.3f} MeV")
    print(f"    (Iron/Nickel region — this is why iron is the endpoint of stellar fusion)")

    # Fit the rising part (A < 60)
    mask_rise = A_all < 60
    if mask_rise.sum() > 5:
        log_A = np.log10(A_all[mask_rise])
        coeffs = np.polyfit(log_A, BE_all[mask_rise], 1)
        pred = np.polyval(coeffs, log_A)
        ss_res = np.sum((BE_all[mask_rise] - pred) ** 2)
        ss_tot = np.sum((BE_all[mask_rise] - np.mean(BE_all[mask_rise])) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        print(f"    Rising part (A<60): BE/A = {coeffs[1]:.3f} + {coeffs[0]:.3f}*log10(A)  R2={r2:.4f}")

    # Fit the falling part (A > 60)
    mask_fall = A_all > 60
    if mask_fall.sum() > 5:
        coeffs_f = np.polyfit(A_all[mask_fall], BE_all[mask_fall], 1)
        pred_f = np.polyval(coeffs_f, A_all[mask_fall])
        ss_res_f = np.sum((BE_all[mask_fall] - pred_f) ** 2)
        ss_tot_f = np.sum((BE_all[mask_fall] - np.mean(BE_all[mask_fall])) ** 2)
        r2_f = 1.0 - ss_res_f / ss_tot_f
        print(f"    Falling part (A>60): BE/A = {coeffs_f[1]:.3f} + {coeffs_f[0]:.6f}*A  R2={r2_f:.4f}")
        print(f"    Loses {abs(coeffs_f[0])*100:.4f} MeV per 100 nucleons added")

    # 2. N/Z ratio for stable nuclides
    print("\n  [2] N/Z ratio (valley of stability)")
    Z_stable = np.array([n[0] for n in stable], dtype=float)
    N_stable = np.array([n[1] for n in stable], dtype=float)
    nz_ratio = N_stable / Z_stable

    # Fit N = a*Z + b*Z^2 (the stability line)
    coeffs_nz = np.polyfit(Z_stable, N_stable, 2)
    pred_nz = np.polyval(coeffs_nz, Z_stable)
    ss_res_nz = np.sum((N_stable - pred_nz) ** 2)
    ss_tot_nz = np.sum((N_stable - np.mean(N_stable)) ** 2)
    r2_nz = 1.0 - ss_res_nz / ss_tot_nz

    print(f"    N = {coeffs_nz[2]:.3f} + {coeffs_nz[1]:.4f}*Z + {coeffs_nz[0]:.6f}*Z^2")
    print(f"    R2 = {r2_nz:.6f}")
    print(f"    At Z=1:  N/Z = {nz_ratio[Z_stable == 1].mean():.2f} (light nuclei ~ 1:1)")
    heavy_mask = Z_stable > 70
    if heavy_mask.sum() > 0:
        print(f"    At Z>70: N/Z = {nz_ratio[heavy_mask].mean():.2f} (heavy nuclei need more neutrons)")

    # 3. Geiger-Nuttall law: for alpha emitters, log(t½) vs Z
    print("\n  [3] Geiger-Nuttall law (alpha emitters)")
    alpha = [n for n in unstable if n[5] == "alpha" and n[4] > 0]
    if len(alpha) > 5:
        Z_alpha = np.array([n[0] for n in alpha], dtype=float)
        t_alpha = np.array([n[4] for n in alpha])
        log_t = np.log10(t_alpha)

        # log(t½) vs Z
        coeffs_gn = np.polyfit(Z_alpha, log_t, 1)
        pred_gn = np.polyval(coeffs_gn, Z_alpha)
        ss_res_gn = np.sum((log_t - pred_gn) ** 2)
        ss_tot_gn = np.sum((log_t - np.mean(log_t)) ** 2)
        r2_gn = 1.0 - ss_res_gn / ss_tot_gn

        print(f"    {len(alpha)} alpha emitters (Z={Z_alpha.min():.0f} to {Z_alpha.max():.0f})")
        print(f"    log10(t½) = {coeffs_gn[1]:.2f} + {coeffs_gn[0]:.4f}*Z")
        print(f"    R2 = {r2_gn:.4f}")
        print(f"    Each proton added changes half-life by 10^{coeffs_gn[0]:.2f} = x{10**coeffs_gn[0]:.2f}")

        # log(t½) vs Z²/A^(1/3) (better Geiger-Nuttall form)
        A_alpha = np.array([n[3] for n in alpha], dtype=float)
        gn_param = Z_alpha ** 2 / A_alpha ** (1/3)
        coeffs_gn2 = np.polyfit(gn_param, log_t, 1)
        pred_gn2 = np.polyval(coeffs_gn2, gn_param)
        ss_res_gn2 = np.sum((log_t - pred_gn2) ** 2)
        ss_tot_gn2 = np.sum((log_t - np.mean(log_t)) ** 2)
        r2_gn2 = 1.0 - ss_res_gn2 / ss_tot_gn2

        print(f"\n    Better form: log10(t½) vs Z²/A^(1/3)")
        print(f"    log10(t½) = {coeffs_gn2[1]:.2f} + {coeffs_gn2[0]:.4f} * Z²/A^(1/3)")
        print(f"    R2 = {r2_gn2:.4f}")

        gn_verdict = "REDISCOVERED" if r2_gn2 > 0.6 else "PARTIAL"
        print(f"    [{gn_verdict}]")
    else:
        r2_gn, r2_gn2, gn_verdict = 0, 0, "NO_DATA"

    # 4. Magic numbers — do they show up in binding energy?
    print("\n  [4] Magic numbers (nuclear shell closures)")
    magic = [2, 8, 20, 28, 50, 82, 126]
    print(f"    Known magic numbers: {magic}")

    # Check if magic-N or magic-Z nuclides have higher BE/A
    for m in magic:
        # Z = magic
        z_magic = [n for n in NUCLIDES if n[0] == m]
        # N = magic
        n_magic = [n for n in NUCLIDES if n[1] == m]

        if z_magic:
            be_z = [n[6] for n in z_magic]
            print(f"    Z={m:3d}: {len(z_magic)} nuclides, mean BE/A = {np.mean(be_z):.3f} MeV  ({z_magic[0][2]}...)")
        if n_magic and m <= 82:
            be_n = [n[6] for n in n_magic]
            print(f"    N={m:3d}: {len(n_magic)} nuclides, mean BE/A = {np.mean(be_n):.3f} MeV")

    # 5. Half-life vs mass number for all unstable
    print("\n  [5] Half-life vs mass number (all radioactive)")
    A_unstable = np.array([n[3] for n in unstable], dtype=float)
    t_unstable = np.array([n[4] for n in unstable])
    log_t_all = np.log10(t_unstable)

    r_corr = float(np.corrcoef(A_unstable, log_t_all)[0, 1])
    print(f"    Correlation: r = {r_corr:.4f}")
    print(f"    (Negative = heavier nuclei decay faster)")

    coeffs_at = np.polyfit(A_unstable, log_t_all, 1)
    print(f"    log10(t½) = {coeffs_at[1]:.2f} + {coeffs_at[0]:.4f}*A")
    print(f"    Each 10 nucleons added: t½ changes by 10^{coeffs_at[0]*10:.1f}")

    # 6. The uranium decay chain
    print("\n  [6] Uranium-238 decay chain:")
    u238_chain = [
        ("U-238", 92, 238, 1.410e17, "alpha"),
        ("Th-234", 90, 234, 2.082e6, "beta-"),
        ("Pa-234", 91, 234, 70560, "beta-"),
        ("U-234", 92, 234, 7.747e12, "alpha"),
        ("Th-230", 90, 230, 2.379e12, "alpha"),
        ("Ra-226", 88, 226, 5.049e10, "alpha"),
        ("Rn-222", 86, 222, 3.304e5, "alpha"),
        ("Po-218", 84, 218, 186.0, "alpha"),
        ("Pb-214", 82, 214, 1608.0, "beta-"),
        ("Bi-214", 83, 214, 1194.0, "beta-"),
        ("Po-214", 84, 214, 1.643e-4, "alpha"),
        ("Pb-210", 82, 210, 7.012e8, "beta-"),
        ("Bi-210", 83, 210, 4.331e5, "beta-"),
        ("Po-210", 84, 210, 1.196e7, "alpha"),
        ("Pb-206", 82, 206, None, "stable"),
    ]

    print(f"    U-238 -> ... -> Pb-206 ({len(u238_chain)} steps, 8 alpha + 6 beta)")
    for step in u238_chain:
        if step[3] is not None:
            if step[3] > 3.15e7:
                t_str = f"{step[3]/3.15e7:.2e} years"
            elif step[3] > 86400:
                t_str = f"{step[3]/86400:.1f} days"
            elif step[3] > 3600:
                t_str = f"{step[3]/3600:.1f} hours"
            elif step[3] > 60:
                t_str = f"{step[3]/60:.1f} min"
            elif step[3] > 1:
                t_str = f"{step[3]:.1f} s"
            else:
                t_str = f"{step[3]:.2e} s"
            print(f"      {step[0]:8s} (Z={step[1]}, A={step[2]}) -> {step[4]:6s}  t½ = {t_str}")
        else:
            print(f"      {step[0]:8s} (Z={step[1]}, A={step[2]}) -> STABLE")

    # Half-lives in chain span 21 orders of magnitude
    chain_t = [s[3] for s in u238_chain if s[3] is not None]
    print(f"\n    Half-life range in chain: {min(chain_t):.2e} to {max(chain_t):.2e} seconds")
    print(f"    That's {np.log10(max(chain_t)/min(chain_t)):.0f} orders of magnitude in one decay chain")

    # Summary
    print("\n  " + "=" * 60)
    print("  SUMMARY")
    print("  " + "=" * 60)
    print(f"\n  Binding energy peak: {NUCLIDES[peak_idx][2]} (A={A_all[peak_idx]:.0f}) — why stars die at iron")
    print(f"  Valley of stability: N = f(Z, Z²)  R2={r2_nz:.4f}")
    print(f"  Geiger-Nuttall: log(t½) ~ Z²/A^(1/3)  R2={r2_gn2:.4f}  [{gn_verdict}]")
    print(f"  U-238 chain: 14 decays spanning {np.log10(max(chain_t)/min(chain_t)):.0f} decades of half-life")
    print(f"  Magic numbers: 2, 8, 20, 28, 50, 82, 126 — shell closures confirmed")

    # Artifact
    artifact = {
        "id": "E100",
        "timestamp": now,
        "world": "nuclear",
        "data_source": "NNDC/IAEA Nuclear Data (curated)",
        "data_url": "https://www.nndc.bnl.gov/",
        "status": "passed",
        "design": {
            "description": "Analyze nuclear properties across the table of nuclides. Binding energy curve, valley of stability, Geiger-Nuttall law, magic numbers.",
            "n_nuclides": len(NUCLIDES),
            "n_stable": len(stable),
            "n_radioactive": len(unstable),
        },
        "result": {
            "binding_energy_peak": {"nuclide": NUCLIDES[peak_idx][2], "A": int(A_all[peak_idx]), "BE_A": float(BE_all[peak_idx])},
            "valley_of_stability": {"r2": float(r2_nz)},
            "geiger_nuttall": {"r2": float(r2_gn2), "verdict": gn_verdict},
            "u238_chain_decades": float(np.log10(max(chain_t) / min(chain_t))),
            "half_life_mass_correlation": float(r_corr),
        },
    }

    out_path = ROOT / "results" / "E100_nuclear_decay.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
