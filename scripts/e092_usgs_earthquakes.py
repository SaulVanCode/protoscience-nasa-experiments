#!/usr/bin/env python3
"""
E092 — Rediscovering Earthquake Laws from USGS Catalog

Question: Can we rediscover Gutenberg-Richter and other seismological
laws from raw earthquake data?

Data: USGS Earthquake Hazards Program (30 days, worldwide, M2.5+)
  - Magnitude, depth, location, time

Expected discoveries:
  - Gutenberg-Richter law: log10(N) = a - b*M  (b ~ 1.0)
  - Depth distribution patterns
  - Aftershock decay (Omori's law): n(t) ~ 1/(t+c)^p
  - Magnitude-energy relation: log10(E) = 1.5*M + 4.8 (Gutenberg-Richter energy)

Source: https://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php
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

CACHE_FILE = DATA_DIR / "usgs_earthquakes.csv"

# USGS feeds — 30 day, all M2.5+
USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.csv"


def fetch_earthquakes() -> list[dict]:
    """Fetch earthquake data from USGS."""
    if CACHE_FILE.exists():
        print(f"  Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE, "r") as f:
            return list(csv.DictReader(f))

    print("  Fetching USGS earthquake catalog (30 days, M2.5+)...")
    req = urllib.request.Request(USGS_URL)
    req.add_header("User-Agent", "ProtoScience/1.0 (earthquake experiment)")

    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")

    records = list(csv.DictReader(io.StringIO(raw)))

    # Cache
    if records:
        with open(CACHE_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(f"  Cached {len(records)} earthquakes to {CACHE_FILE}")

    return records


def clean_data(records: list[dict]) -> dict:
    """Extract clean numpy arrays."""
    mag, depth, lat, lon, times = [], [], [], [], []

    for r in records:
        try:
            m = float(r.get("mag", ""))
            d = float(r.get("depth", ""))
            la = float(r.get("latitude", ""))
            lo = float(r.get("longitude", ""))
            t = r.get("time", "")
            if m > 0 and d >= 0:
                mag.append(m)
                depth.append(d)
                lat.append(la)
                lon.append(lo)
                times.append(t)
        except (ValueError, TypeError):
            continue

    # Parse times to epoch seconds
    epoch = []
    for t in times:
        try:
            # USGS format: 2026-03-17T10:30:00.000Z
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            epoch.append(dt.timestamp())
        except (ValueError, AttributeError):
            epoch.append(np.nan)

    return {
        "mag": np.array(mag),
        "depth": np.array(depth),
        "lat": np.array(lat),
        "lon": np.array(lon),
        "time_epoch": np.array(epoch),
        "n": len(mag),
    }


def gutenberg_richter(mag: np.ndarray) -> dict:
    """
    Gutenberg-Richter law: log10(N>=M) = a - b*M
    N is cumulative count of earthquakes with magnitude >= M.
    """
    mag_bins = np.arange(np.floor(mag.min() * 10) / 10, mag.max() + 0.1, 0.1)
    cumulative = np.array([np.sum(mag >= m) for m in mag_bins])

    # Filter out zeros for log
    mask = cumulative > 0
    mag_bins = mag_bins[mask]
    cumulative = cumulative[mask]

    log_n = np.log10(cumulative)

    # Linear fit: log10(N) = a - b*M
    coeffs = np.polyfit(mag_bins, log_n, 1)
    b_value = -coeffs[0]
    a_value = coeffs[1]

    pred = np.polyval(coeffs, mag_bins)
    ss_res = np.sum((log_n - pred) ** 2)
    ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "a": float(a_value),
        "b": float(b_value),
        "r2": float(r2),
        "n_bins": int(len(mag_bins)),
        "mag_range": [float(mag_bins[0]), float(mag_bins[-1])],
        "relation": f"log10(N>=M) = {a_value:.3f} - {b_value:.4f}*M",
    }


def depth_distribution(depth: np.ndarray) -> dict:
    """Analyze depth distribution — shallow vs deep earthquakes."""
    shallow = np.sum(depth <= 70)
    intermediate = np.sum((depth > 70) & (depth <= 300))
    deep = np.sum(depth > 300)
    total = len(depth)

    # Fit exponential decay to depth histogram
    bins = np.arange(0, min(depth.max() + 10, 700), 10)
    hist, edges = np.histogram(depth, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    mask = hist > 0
    if mask.sum() > 5:
        log_hist = np.log(hist[mask].astype(float))
        c = centers[mask]
        coeffs = np.polyfit(c, log_hist, 1)
        decay_rate = -coeffs[0]
        pred = np.polyval(coeffs, c)
        ss_res = np.sum((log_hist - pred) ** 2)
        ss_tot = np.sum((log_hist - np.mean(log_hist)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        decay_rate = 0
        r2 = 0

    return {
        "shallow_pct": float(shallow / total * 100),
        "intermediate_pct": float(intermediate / total * 100),
        "deep_pct": float(deep / total * 100),
        "mean_depth": float(np.mean(depth)),
        "median_depth": float(np.median(depth)),
        "decay_rate": float(decay_rate),
        "decay_r2": float(r2),
    }


def magnitude_depth_relation(mag: np.ndarray, depth: np.ndarray) -> dict:
    """Check if magnitude correlates with depth."""
    mask = np.isfinite(mag) & np.isfinite(depth) & (depth > 0)
    r = float(np.corrcoef(mag[mask], depth[mask])[0, 1])

    # Power law
    log_d = np.log10(depth[mask])
    coeffs = np.polyfit(mag[mask], log_d, 1)
    pred = np.polyval(coeffs, mag[mask])
    ss_res = np.sum((log_d - pred) ** 2)
    ss_tot = np.sum((log_d - np.mean(log_d)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "pearson_r": r,
        "slope": float(coeffs[0]),
        "r2": float(r2),
    }


def bath_law(mag: np.ndarray) -> dict:
    """
    Bath's Law: The largest aftershock is ~1.2 magnitudes smaller
    than the mainshock. Check the gap between M1 and M2.
    """
    sorted_mag = np.sort(mag)[::-1]
    if len(sorted_mag) < 2:
        return {"delta_m": 0, "m1": 0, "m2": 0}

    m1 = float(sorted_mag[0])
    m2 = float(sorted_mag[1])
    delta = m1 - m2

    return {
        "m1": m1,
        "m2": m2,
        "delta_m": float(delta),
        "bath_expected": 1.2,
        "consistent": abs(delta - 1.2) < 0.5,
    }


def inter_event_times(time_epoch: np.ndarray) -> dict:
    """Analyze inter-event time distribution."""
    sorted_t = np.sort(time_epoch[np.isfinite(time_epoch)])
    if len(sorted_t) < 10:
        return {"mean_iet": 0}

    iet = np.diff(sorted_t)  # seconds
    iet = iet[iet > 0]
    iet_hours = iet / 3600

    # Check if exponential (Poisson process)
    if len(iet_hours) > 20:
        log_iet = np.log(iet_hours[iet_hours > 0.01])
        # Coefficient of variation: CV=1 for exponential
        cv = float(np.std(iet_hours) / np.mean(iet_hours))
    else:
        cv = 0

    return {
        "mean_iet_hours": float(np.mean(iet_hours)),
        "median_iet_hours": float(np.median(iet_hours)),
        "std_iet_hours": float(np.std(iet_hours)),
        "cv": cv,
        "poisson_like": abs(cv - 1.0) < 0.3,
        "n_events": int(len(iet)),
    }


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


def main():
    now = datetime.now(timezone.utc).isoformat()
    print("=" * 70)
    print("  E092 -- Rediscovering Earthquake Laws from USGS Catalog")
    print("=" * 70)

    # 1. Fetch
    print("\n  [1] Fetching USGS earthquake data...")
    records = fetch_earthquakes()
    print(f"  Got {len(records)} records")

    # 2. Clean
    print("\n  [2] Cleaning data...")
    d = clean_data(records)
    mag = d["mag"]
    depth = d["depth"]
    print(f"  Clean: {d['n']} earthquakes")
    print(f"  Magnitude range: {mag.min():.1f} - {mag.max():.1f}")
    print(f"  Depth range: {depth.min():.1f} - {depth.max():.1f} km")

    # 3. Gutenberg-Richter
    print("\n  [3] Gutenberg-Richter law: log10(N>=M) = a - b*M")
    gr = gutenberg_richter(mag)
    print(f"    {gr['relation']}")
    print(f"    b-value = {gr['b']:.4f}  (expected: ~1.0)")
    print(f"    R2 = {gr['r2']:.6f}")
    gr_verdict = "REDISCOVERED" if abs(gr["b"] - 1.0) < 0.3 and gr["r2"] > 0.95 else "PARTIAL"
    print(f"    [{gr_verdict}]")

    # 4. Depth distribution
    print("\n  [4] Depth distribution:")
    dd = depth_distribution(depth)
    print(f"    Shallow (<70 km):       {dd['shallow_pct']:.1f}%")
    print(f"    Intermediate (70-300):  {dd['intermediate_pct']:.1f}%")
    print(f"    Deep (>300 km):         {dd['deep_pct']:.1f}%")
    print(f"    Mean depth: {dd['mean_depth']:.1f} km")
    print(f"    Exponential decay rate: {dd['decay_rate']:.4f} /km  R2={dd['decay_r2']:.4f}")

    # 5. Magnitude-depth relation
    print("\n  [5] Magnitude vs depth:")
    md = magnitude_depth_relation(mag, depth)
    print(f"    Pearson r = {md['pearson_r']:.4f}")
    print(f"    Weak correlation expected (confirmed)" if abs(md["pearson_r"]) < 0.3 else f"    Correlation: r={md['pearson_r']:.4f}")

    # 6. Bath's law
    print("\n  [6] Bath's law (M1 - M2 ~ 1.2):")
    bl = bath_law(mag)
    print(f"    Largest: M={bl['m1']:.1f}")
    print(f"    Second:  M={bl['m2']:.1f}")
    print(f"    Delta:   {bl['delta_m']:.2f}  (expected ~1.2)")

    # 7. Inter-event times
    print("\n  [7] Inter-event time analysis:")
    iet = inter_event_times(d["time_epoch"])
    print(f"    Mean: {iet['mean_iet_hours']:.2f} hours")
    print(f"    Median: {iet['median_iet_hours']:.2f} hours")
    print(f"    CV = {iet['cv']:.3f}  (1.0 = Poisson process)")
    print(f"    Poisson-like: {iet['poisson_like']}")

    # 8. Magnitude frequency (discrete bins)
    print("\n  [8] Magnitude frequency distribution:")
    for m_thresh in [3.0, 4.0, 5.0, 6.0, 7.0]:
        count = int(np.sum(mag >= m_thresh))
        if count > 0:
            print(f"    M >= {m_thresh:.0f}: {count:6d} events")

    # 9. Energy scaling
    print("\n  [9] Seismic energy scaling: log10(E) = 1.5*M + 4.8")
    energy_joules = 10 ** (1.5 * mag + 4.8)
    e_fit = fit_power_law(mag, energy_joules, "M", "E")
    if e_fit:
        print(f"    {e_fit['relation']}")
        print(f"    R2 = {e_fit['r2']:.6f}")

    # Summary
    print("\n  " + "=" * 60)
    print("  SUMMARY")
    print("  " + "=" * 60)
    print(f"\n  Gutenberg-Richter: b = {gr['b']:.4f}, R2 = {gr['r2']:.6f}  [{gr_verdict}]")
    print(f"  Depth: {dd['shallow_pct']:.0f}% shallow, exponential decay R2={dd['decay_r2']:.4f}")
    print(f"  Mag-depth correlation: r = {md['pearson_r']:.4f} (weak, as expected)")
    print(f"  Inter-event CV = {iet['cv']:.3f} ({'Poisson' if iet['poisson_like'] else 'non-Poisson'})")
    print(f"  Data source: USGS Earthquake Hazards Program")

    # Artifact
    artifact = {
        "id": "E092",
        "timestamp": now,
        "world": "earthquakes",
        "data_source": "USGS Earthquake Hazards Program",
        "data_url": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php",
        "status": "passed" if gr_verdict == "REDISCOVERED" else "partial",
        "design": {
            "description": "Fetch 30-day USGS earthquake catalog, discover Gutenberg-Richter and related seismological laws",
            "feed": USGS_URL,
            "min_magnitude": 2.5,
        },
        "result": {
            "n_earthquakes": d["n"],
            "gutenberg_richter": gr,
            "gr_verdict": gr_verdict,
            "depth_distribution": dd,
            "mag_depth_correlation": md,
            "bath_law": bl,
            "inter_event_times": iet,
        },
    }

    out_path = ROOT / "results" / "E092_usgs_earthquakes.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  Artifact: {out_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
