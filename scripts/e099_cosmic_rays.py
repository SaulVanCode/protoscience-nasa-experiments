#!/usr/bin/env python3
"""
E099 — Cosmic Ray Energy Spectrum: The Knee, the Ankle, and Beyond

Question: Can we rediscover the power-law structure of the cosmic ray
spectrum, including the famous "knee" and "ankle" transitions?

Background:
  Cosmic rays bombard Earth from space with energies spanning 12 orders
  of magnitude. The flux follows a power law J ~ E^(-gamma) but with
  TWO breaks:
  - The "knee" at ~3×10^15 eV: gamma changes from 2.7 to 3.1
  - The "ankle" at ~5×10^18 eV: gamma changes from 3.1 to 2.6
  - The GZK cutoff at ~5×10^19 eV: flux drops sharply (protons
    interact with CMB photons)

  Nobody fully understands why these breaks exist. The knee may be
  where galactic accelerators (supernovae) reach their limit. The
  ankle may be where extragalactic sources take over.

Data: Published spectrum measurements compiled from multiple experiments
  (Auger, Telescope Array, KASCADE, Tibet, IceTop)

Source: PDG cosmic ray review + published Auger/TA results
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# ── Cosmic Ray Spectrum Data ────────────────────────────────────
# Energy (eV) vs Flux J (particles / m² sr s GeV)
# Compiled from PDG review, Auger, Telescope Array, KASCADE
# Flux is differential: dN/dE
# These are approximate central values from published figures

SPECTRUM = [
    # (log10(E/eV), log10(J / [m^-2 sr^-1 s^-1 GeV^-1]))
    # Low energy (direct measurements, satellites)
    (9.0,   3.0),
    (9.5,   1.7),
    (10.0,  0.5),
    (10.5, -0.8),
    (11.0, -2.0),
    (11.5, -3.2),
    (12.0, -4.3),
    (12.5, -5.4),
    (13.0, -6.5),
    (13.5, -7.6),
    (14.0, -8.6),
    (14.5, -9.6),
    # The KNEE region (~3×10^15 eV)
    (15.0, -10.7),
    (15.3, -11.3),  # knee starts
    (15.5, -11.8),  # knee
    (15.8, -12.5),
    (16.0, -13.0),
    (16.5, -14.2),
    (17.0, -15.5),
    (17.5, -16.7),
    # The ANKLE region (~5×10^18 eV)
    (18.0, -17.8),
    (18.3, -18.5),
    (18.5, -18.9),  # ankle
    (18.7, -19.2),
    (19.0, -19.6),
    (19.3, -20.2),
    # GZK suppression
    (19.5, -20.8),
    (19.7, -21.5),
    (20.0, -22.5),
    (20.3, -24.0),
]

# Published spectral indices by region
KNOWN_INDICES = {
    "below_knee": {"range": (10.0, 15.0), "gamma": 2.7, "tolerance": 0.2},
    "knee_to_ankle": {"range": (16.0, 18.3), "gamma": 3.1, "tolerance": 0.3},
    "above_ankle": {"range": (18.5, 19.5), "gamma": 2.6, "tolerance": 0.3},
}


def fit_power_law_segment(log_E, log_J, E_min, E_max):
    """Fit log(J) = a - gamma * log(E) in a given energy range."""
    mask = (log_E >= E_min) & (log_E <= E_max)
    if mask.sum() < 3:
        return None

    x = log_E[mask]
    y = log_J[mask]

    coeffs = np.polyfit(x, y, 1)
    gamma = -coeffs[0]  # negative because J decreases with E
    intercept = coeffs[1]

    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "gamma": float(gamma),
        "intercept": float(intercept),
        "r2": float(r2),
        "n": int(mask.sum()),
        "E_range": [float(E_min), float(E_max)],
    }


def find_breaks(log_E, log_J, window=5):
    """Find spectral breaks by looking for changes in local slope."""
    breaks = []
    n = len(log_E)

    for i in range(window, n - window):
        # Fit slope before and after point i
        x_before = log_E[i-window:i]
        y_before = log_J[i-window:i]
        x_after = log_E[i:i+window]
        y_after = log_J[i:i+window]

        if len(x_before) < 3 or len(x_after) < 3:
            continue

        slope_before = np.polyfit(x_before, y_before, 1)[0]
        slope_after = np.polyfit(x_after, y_after, 1)[0]

        delta_slope = slope_after - slope_before

        breaks.append({
            "log_E": float(log_E[i]),
            "E_eV": float(10 ** log_E[i]),
            "slope_before": float(-slope_before),  # gamma
            "slope_after": float(-slope_after),
            "delta_gamma": float(-delta_slope),
        })

    return breaks


def overall_fit(log_E, log_J):
    """Fit the entire spectrum as a single power law."""
    coeffs = np.polyfit(log_E, log_J, 1)
    gamma = -coeffs[0]
    pred = np.polyval(coeffs, log_E)
    ss_res = np.sum((log_J - pred) ** 2)
    ss_tot = np.sum((log_J - np.mean(log_J)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"gamma": float(gamma), "r2": float(r2)}


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E099 -- Cosmic Ray Energy Spectrum")
    print("=" * 70)

    data = np.array(SPECTRUM)
    log_E = data[:, 0]
    log_J = data[:, 1]

    print(f"\n  Data: {len(SPECTRUM)} spectrum points")
    print(f"  Energy range: 10^{log_E[0]:.0f} to 10^{log_E[-1]:.0f} eV")
    print(f"  Flux range: 10^{log_J[0]:.0f} to 10^{log_J[-1]:.0f} (particles/m2/sr/s/GeV)")
    print(f"  Spans {log_E[-1] - log_E[0]:.0f} decades in energy")

    # 1. Overall fit
    print("\n  [1] Overall power law (single slope):")
    overall = overall_fit(log_E, log_J)
    print(f"    J ~ E^(-{overall['gamma']:.4f})  R2={overall['r2']:.6f}")
    print(f"    A single power law explains {overall['r2']*100:.1f}% of the variance")

    # 2. Segmented fits
    print("\n  [2] Segmented power laws:")
    segments = {}
    for name, spec in KNOWN_INDICES.items():
        E_lo, E_hi = spec["range"]
        fit = fit_power_law_segment(log_E, log_J, E_lo, E_hi)
        if fit:
            segments[name] = fit
            err = abs(fit["gamma"] - spec["gamma"])
            verdict = "MATCH" if err < spec["tolerance"] else "MISS"
            print(f"\n    {name} (10^{E_lo:.0f} - 10^{E_hi:.0f} eV):")
            print(f"      gamma = {fit['gamma']:.4f}  (expected: {spec['gamma']:.1f}, err: {err:.4f})")
            print(f"      R2 = {fit['r2']:.6f}  n={fit['n']}")
            print(f"      [{verdict}]")

    # 3. Break detection
    print("\n  [3] Spectral break detection:")
    breaks = find_breaks(log_E, log_J, window=4)

    # Find the two most significant breaks
    if breaks:
        breaks.sort(key=lambda b: -abs(b["delta_gamma"]))
        print(f"    Found {len(breaks)} slope change points")
        print(f"\n    Top spectral breaks:")
        for i, b in enumerate(breaks[:5]):
            E_label = ""
            if 14.5 < b["log_E"] < 16.0:
                E_label = " <-- KNEE?"
            elif 18.0 < b["log_E"] < 19.0:
                E_label = " <-- ANKLE?"
            elif b["log_E"] > 19.3:
                E_label = " <-- GZK?"
            print(f"      {i+1}. E = 10^{b['log_E']:.1f} eV  delta_gamma = {b['delta_gamma']:+.4f}  (gamma: {b['slope_before']:.2f} -> {b['slope_after']:.2f}){E_label}")

    # 4. The knee
    print("\n  [4] The Knee (~3x10^15 eV):")
    knee_before = fit_power_law_segment(log_E, log_J, 11.0, 14.5)
    knee_after = fit_power_law_segment(log_E, log_J, 15.5, 17.5)
    if knee_before and knee_after:
        delta = knee_after["gamma"] - knee_before["gamma"]
        print(f"    Before knee: gamma = {knee_before['gamma']:.4f}  (R2={knee_before['r2']:.4f})")
        print(f"    After knee:  gamma = {knee_after['gamma']:.4f}  (R2={knee_after['r2']:.4f})")
        print(f"    Delta gamma = {delta:+.4f}")
        print(f"    Expected: ~+0.4 (2.7 -> 3.1)")
        knee_verdict = "REDISCOVERED" if 0.2 < delta < 0.8 else "PARTIAL"
        print(f"    [{knee_verdict}]")
    else:
        knee_verdict = "NO_DATA"

    # 5. The ankle
    print("\n  [5] The Ankle (~5x10^18 eV):")
    ankle_before = fit_power_law_segment(log_E, log_J, 16.5, 18.3)
    ankle_after = fit_power_law_segment(log_E, log_J, 18.5, 19.5)
    if ankle_before and ankle_after:
        delta = ankle_after["gamma"] - ankle_before["gamma"]
        print(f"    Before ankle: gamma = {ankle_before['gamma']:.4f}  (R2={ankle_before['r2']:.4f})")
        print(f"    After ankle:  gamma = {ankle_after['gamma']:.4f}  (R2={ankle_after['r2']:.4f})")
        print(f"    Delta gamma = {delta:+.4f}")
        print(f"    Expected: ~-0.5 (3.1 -> 2.6)")
        ankle_verdict = "REDISCOVERED" if -0.8 < delta < -0.1 else "PARTIAL"
        print(f"    [{ankle_verdict}]")
    else:
        ankle_verdict = "NO_DATA"

    # 6. Energy budget
    print("\n  [6] Cosmic ray energy facts:")
    # One particle per m² per second at 10^9 eV
    # One particle per km² per year at 10^19 eV
    print(f"    At 10^9 eV:  ~1000 particles/m2/s  (constant rain)")
    print(f"    At 10^15 eV: ~1 particle/m2/year   (knee)")
    print(f"    At 10^19 eV: ~1 particle/km2/year  (ankle)")
    print(f"    At 10^20 eV: ~1 particle/km2/century (extreme)")
    print(f"    Highest ever: 3x10^20 eV ('Oh-My-God' particle, 1991)")
    print(f"    That's a tennis ball at 100 km/h — in one proton.")

    # Summary
    print("\n  " + "=" * 60)
    print("  SUMMARY")
    print("  " + "=" * 60)

    n_rediscovered = sum(1 for v in [knee_verdict, ankle_verdict] if v == "REDISCOVERED")
    print(f"\n  Overall spectrum: J ~ E^(-{overall['gamma']:.2f})  R2={overall['r2']:.4f}")
    print(f"  Knee (10^15 eV): [{knee_verdict}]")
    print(f"  Ankle (10^18.5 eV): [{ankle_verdict}]")
    print(f"  Score: {n_rediscovered}/2 spectral features identified")

    if segments:
        print(f"\n  Spectral indices by region:")
        for name, s in segments.items():
            print(f"    {name:20s}: gamma = {s['gamma']:.3f}  R2={s['r2']:.4f}")

    # Artifact
    artifact = {
        "id": "E099",
        "timestamp": now,
        "world": "cosmic_rays",
        "data_source": "PDG Review + Pierre Auger + Telescope Array (compiled)",
        "data_url": "https://pdg.lbl.gov/",
        "status": "passed" if n_rediscovered >= 1 else "partial",
        "design": {
            "description": "Analyze the cosmic ray energy spectrum for power-law structure, spectral breaks (knee, ankle), and the GZK cutoff",
            "n_points": len(SPECTRUM),
            "energy_range": "10^9 to 10^20.3 eV",
        },
        "result": {
            "overall_fit": overall,
            "segments": segments,
            "knee_verdict": knee_verdict,
            "ankle_verdict": ankle_verdict,
            "n_rediscovered": n_rediscovered,
            "top_breaks": breaks[:5] if breaks else [],
        },
    }

    out_path = ROOT / "results" / "E099_cosmic_rays.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
