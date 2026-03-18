#!/usr/bin/env python3
"""
E107 — Milankovitch Cycles in Antarctic Ice Cores

Question: Can we rediscover the orbital frequencies that control
Earth's ice ages from 420,000 years of temperature data?

Background:
  Milutin Milankovitch (1920s) proposed that ice ages are driven
  by slow changes in Earth's orbit:

  1. Eccentricity (~100,000 year cycle): Earth's orbit goes from
     nearly circular to slightly elliptical. Controlled by Jupiter
     and Saturn's gravitational pull.

  2. Obliquity (~41,000 year cycle): Earth's axial tilt varies
     between 22.1° and 24.5°. More tilt = stronger seasons.

  3. Precession (~23,000 year cycle): Earth's axis wobbles like
     a top. Changes which hemisphere gets more summer sun.

  These cycles were confirmed in the 1970s by Hays, Imbrie & Shackleton
  using ocean sediment cores. The Vostok ice core (1999) provided the
  longest continuous temperature record from a single location.

Data: Vostok Ice Core deuterium-derived temperature record
  3,310 data points spanning 420,000 years

Source: Petit et al. (1999), Nature 399, 429-436
  https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/
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

CACHE_FILE = DATA_DIR / "vostok_temperature.txt"

VOSTOK_URL = "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/deutnat.txt"

# Known Milankovitch periods (years)
KNOWN_PERIODS = {
    "Eccentricity": 100000,
    "Obliquity": 41000,
    "Precession": 23000,
}


def fetch_vostok() -> tuple:
    """Fetch Vostok ice core temperature data."""
    if CACHE_FILE.exists():
        print(f"  Loading cached data from {CACHE_FILE}")
        raw = CACHE_FILE.read_text(encoding="utf-8", errors="replace")
    else:
        print("  Fetching Vostok ice core data from NOAA...")
        req = urllib.request.Request(VOSTOK_URL)
        req.add_header("User-Agent", "ProtoScience/1.0 (paleoclimate experiment)")
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        CACHE_FILE.write_text(raw, encoding="utf-8")
        print(f"  Cached to {CACHE_FILE}")

    # Parse — skip header lines, find the data
    ages = []
    temps = []
    depths = []

    in_data = False
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Try to parse as data (4 numbers)
        parts = line.split()
        if len(parts) >= 4:
            try:
                depth = float(parts[0])
                age = float(parts[1])
                deut = float(parts[2])
                delta_t = float(parts[3])

                if 0 < age < 500000 and -15 < delta_t < 15:
                    ages.append(age)
                    temps.append(delta_t)
                    depths.append(depth)
                    in_data = True
            except ValueError:
                continue

    return np.array(ages), np.array(temps), np.array(depths)


def compute_fft(ages, temps):
    """Compute FFT on irregularly sampled data using interpolation."""
    # Interpolate to regular grid
    age_min, age_max = ages.min(), ages.max()
    n_interp = 4096  # power of 2 for FFT
    age_regular = np.linspace(age_min, age_max, n_interp)
    temp_regular = np.interp(age_regular, np.sort(ages), temps[np.argsort(ages)])

    # Remove mean
    temp_regular -= np.mean(temp_regular)

    # Apply Hanning window
    window = np.hanning(n_interp)
    temp_windowed = temp_regular * window

    # FFT
    fft_vals = np.fft.rfft(temp_windowed)
    freqs = np.fft.rfftfreq(n_interp, d=(age_max - age_min) / n_interp)

    # Power spectrum
    power = np.abs(fft_vals) ** 2
    periods = 1.0 / freqs[1:]  # skip DC component
    power = power[1:]

    return periods, power, age_regular, temp_regular


def find_peaks(periods, power, min_period=5000, max_period=200000, n_peaks=10):
    """Find the strongest periodic signals."""
    mask = (periods >= min_period) & (periods <= max_period)
    p = periods[mask]
    pw = power[mask]

    # Find local maxima
    peaks = []
    for i in range(1, len(pw) - 1):
        if pw[i] > pw[i-1] and pw[i] > pw[i+1]:
            peaks.append((p[i], pw[i]))

    # Sort by power
    peaks.sort(key=lambda x: -x[1])
    return peaks[:n_peaks]


def match_milankovitch(peaks):
    """Match found peaks to known Milankovitch frequencies."""
    matches = {}
    for name, known_period in KNOWN_PERIODS.items():
        best_match = None
        best_err = float("inf")
        for period, power in peaks:
            err = abs(period - known_period) / known_period
            if err < best_err:
                best_err = err
                best_match = (period, power, err)

        if best_match and best_err < 0.25:  # within 25%
            matches[name] = {
                "known_period": known_period,
                "found_period": float(best_match[0]),
                "power": float(best_match[1]),
                "error_pct": float(best_match[2] * 100),
                "matched": bool(best_err < 0.15),
            }
        else:
            matches[name] = {
                "known_period": known_period,
                "found_period": None,
                "matched": False,  # plain Python bool
            }

    return matches


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E107 -- Milankovitch Cycles in Antarctic Ice Cores")
    print("=" * 70)

    # 1. Fetch data
    print(f"\n  [1] Fetching Vostok ice core data...")
    ages, temps, depths = fetch_vostok()
    print(f"  {len(ages)} data points")
    print(f"  Time span: {ages.min():.0f} to {ages.max():.0f} years BP")
    print(f"  Temperature range: {temps.min():.1f} to {temps.max():.1f} C (relative to recent)")
    print(f"  Depth range: {depths.min():.0f} to {depths.max():.0f} meters")

    # 2. Basic statistics
    print(f"\n  [2] Temperature statistics:")
    print(f"    Mean anomaly: {np.mean(temps):.2f} C")
    print(f"    Std: {np.std(temps):.2f} C")
    print(f"    Most recent ~10,000 years are WARM (interglacial)")
    print(f"    The other ~90% of the time was ICE AGE")

    # Count glacial cycles
    threshold = -2.0  # below this = glacial
    glacial = temps < threshold
    transitions = np.diff(glacial.astype(int))
    n_cycles = np.sum(transitions == 1)
    print(f"    Number of glacial cycles: ~{n_cycles}")
    print(f"    Average cycle length: ~{int(ages.max() / max(n_cycles, 1))} years")

    # 3. FFT
    print(f"\n  [3] Fourier Transform (searching for periodic signals)...")
    periods, power, age_reg, temp_reg = compute_fft(ages, temps)

    # Find peaks
    peaks = find_peaks(periods, power)
    print(f"\n  Top 10 periodic signals:")
    print(f"  {'Rank':>5s} {'Period (yr)':>12s} {'Power':>12s}")
    print(f"  {'-'*5} {'-'*12} {'-'*12}")
    for i, (period, pw) in enumerate(peaks):
        label = ""
        for name, known in KNOWN_PERIODS.items():
            if abs(period - known) / known < 0.15:
                label = f" <-- {name}!"
        print(f"  {i+1:5d} {period:12.0f} {pw:12.1f}{label}")

    # 4. Match to Milankovitch
    print(f"\n  [4] Matching to known Milankovitch frequencies:")
    matches = match_milankovitch(peaks)

    n_matched = 0
    for name, m in matches.items():
        if m["matched"]:
            n_matched += 1
            print(f"    [OK] {name:15s}: expected {m['known_period']:>7,d} yr, found {m['found_period']:>7,.0f} yr (error: {m['error_pct']:.1f}%)")
        elif m["found_period"]:
            print(f"    [~~] {name:15s}: expected {m['known_period']:>7,d} yr, found {m['found_period']:>7,.0f} yr (error: {m['error_pct']:.1f}%)")
        else:
            print(f"    [NO] {name:15s}: expected {m['known_period']:>7,d} yr, not found")

    # 5. Temperature-CO2 connection
    print(f"\n  [5] The ice age pattern:")
    print(f"    Earth's temperature over 420,000 years shows:")
    print(f"    - 4 major ice ages (glacial periods)")
    print(f"    - Brief warm periods (interglacials) lasting ~10-15k years")
    print(f"    - We are currently in an interglacial (Holocene)")
    print(f"    - Without human intervention, the next ice age would begin")
    print(f"      in roughly 50,000 years")

    # 6. The dominant frequency
    if peaks:
        dominant = peaks[0]
        print(f"\n  [6] Dominant cycle: {dominant[0]:,.0f} years")
        if abs(dominant[0] - 100000) / 100000 < 0.2:
            print(f"    This is the ~100,000-year eccentricity cycle")
            print(f"    Controlled by Jupiter and Saturn's gravity!")
            print(f"    The biggest planets in the solar system control")
            print(f"    when Earth has ice ages.")

    # Summary
    print(f"\n  " + "=" * 60)
    print(f"  SUMMARY")
    print(f"  " + "=" * 60)
    verdict = "REDISCOVERED" if n_matched >= 2 else "PARTIAL"
    print(f"\n  Milankovitch cycles: {n_matched}/3 frequencies identified [{verdict}]")
    print(f"  Data: {len(ages)} points spanning {ages.max():,.0f} years")
    print(f"  Source: Vostok ice core, Antarctica")
    print(f"\n  Earth's ice ages are controlled by orbital mechanics.")
    print(f"  Jupiter, 778 million km away, determines whether")
    print(f"  Chicago is under 2 km of ice or not.")

    # Artifact
    artifact = {
        "id": "E107",
        "timestamp": now,
        "world": "paleoclimate",
        "data_source": "Vostok Ice Core (Petit et al. 1999)",
        "data_url": "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/",
        "status": "passed" if verdict == "REDISCOVERED" else "partial",
        "design": {
            "description": "Apply FFT to 420,000 years of Antarctic ice core temperature data to find Milankovitch orbital cycles",
            "n_points": len(ages),
            "time_span_years": float(ages.max()),
        },
        "result": {
            "n_glacial_cycles": int(n_cycles),
            "top_peaks": [{"period": float(p), "power": float(pw)} for p, pw in peaks[:10]],
            "milankovitch_matches": matches,
            "n_matched": n_matched,
            "verdict": verdict,
            "dominant_cycle_years": float(peaks[0][0]) if peaks else 0,
        },
    }

    out_path = ROOT / "results" / "E107_milankovitch.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
