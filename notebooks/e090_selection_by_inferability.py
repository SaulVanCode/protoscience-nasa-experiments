"""
E090 — Selection by Inferability: Phase Transitions in Discoverability

Can an agent discover the laws of its universe? Does the answer depend on
the universe's structure? We simulate 1,460 toy universes and measure when
predictive modeling succeeds or fails.

Result: sharp phase transitions in discoverability. Below a critical noise
threshold, agents discover laws in ~44 ticks. Above it, never. The boundary
is razor-thin (delta ~ 0.015).

Full code: https://github.com/SaulVanCode/protoscience-nasa-experiments
Simulation engine: world-inference/ (deterministic, seeded, append-only ledger)

Run: python e090_selection_by_inferability.py
"""

import json
import sys
import os

# ── Results from study_v01 (1,460 runs, 20 seeds per param) ────────

RESULTS = {
    "simple": {
        "description": "Linear AR(2) with additive noise",
        "law": "s(t+1) = 0.7*s(t) + 0.2*s(t-1) + 0.1 + noise",
        "parameter": "noise_std",
        "transition": {
            "reliable_discovery": "noise_std <= 0.090",
            "total_collapse": "noise_std >= 0.120",
            "transition_width": 0.015,
            "D_at_zero_noise": 0.998,
            "D_at_transition": 0.31,
        },
        "curve": [
            (0.000, 1.00, 0.998, 44),
            (0.023, 1.00, 0.716, 44),
            (0.047, 1.00, 0.525, 50),
            (0.070, 1.00, 0.395, 79),
            (0.080, 1.00, 0.358, 135),
            (0.090, 0.95, 0.315, 195),
            (0.095, 0.75, 0.262, 242),
            (0.100, 0.40, 0.218, 219),
            (0.105, 0.20, 0.183, 265),
            (0.110, 0.10, 0.173, 198),
            (0.120, 0.00, 0.157, None),
            (0.150, 0.00, 0.157, None),
        ],
    },
    "structured": {
        "description": "AR(1) + hidden periodic driver (period=20) + noise",
        "law": "s(t+1) = 0.5*s(t) + 0.3*sin(2*pi*t/20) + noise",
        "parameter": "noise_std",
        "transition": {
            "reliable_discovery": "noise_std <= 0.085",
            "total_collapse": "noise_std >= 0.120",
            "transition_width": 0.020,
            "D_at_zero_noise": 0.998,
            "D_at_transition": 0.30,
        },
        "curve": [
            (0.000, 1.00, 0.998, 44),
            (0.018, 1.00, 0.752, 44),
            (0.035, 1.00, 0.603, 46),
            (0.050, 1.00, 0.480, 53),
            (0.065, 1.00, 0.396, 83),
            (0.080, 0.95, 0.333, 163),
            (0.085, 0.85, 0.297, 201),
            (0.090, 0.65, 0.242, 263),
            (0.095, 0.40, 0.210, 265),
            (0.100, 0.30, 0.188, 321),
            (0.105, 0.20, 0.178, 320),
            (0.120, 0.00, 0.159, None),
        ],
    },
    "chaotic": {
        "description": "Logistic map s(t+1) = r*s(t)*(1-s(t))",
        "law": "s(t+1) = r * s(t) * (1 - s(t))",
        "parameter": "r",
        "transition": {
            "reliable_discovery": "r <= 3.650",
            "total_collapse": "r >= 3.680",
            "transition_width": 0.070,
            "D_at_ordered": 0.998,
            "D_at_transition": 0.35,
            "periodic_windows": "r=3.63 (D=0.90), r=3.74 (D=0.81)",
        },
        "curve": [
            (2.800, 1.00, 0.998, 44),
            (3.200, 1.00, 0.991, 44),
            (3.500, 1.00, 0.903, 44),
            (3.570, 1.00, 0.814, 44),
            (3.590, 1.00, 0.618, 44),
            (3.610, 1.00, 0.496, 48),
            (3.630, 1.00, 0.904, 48),  # periodic window!
            (3.650, 1.00, 0.353, 168),
            (3.660, 0.40, 0.248, 141),
            (3.670, 0.10, 0.195, 212),
            (3.680, 0.00, 0.174, None),
            (3.740, 1.00, 0.812, 105),  # periodic window!
            (3.800, 0.00, 0.175, None),
            (4.000, 0.00, 0.189, None),
        ],
    },
}


def print_results():
    print("=" * 70)
    print("  E090: SELECTION BY INFERABILITY")
    print("  Phase Transitions in Discoverability")
    print("=" * 70)

    print("\n  1,460 simulations | 3 universe types | 20 seeds per config")
    print("  Agent: linear AR(10) | Discovery threshold: 0.05")
    print("  All runs deterministic and reproducible (seeded RNG)")

    for utype, data in RESULTS.items():
        t = data["transition"]
        print(f"\n  --- {utype.upper()} ---")
        print(f"  {data['description']}")
        print(f"  Law: {data['law']}")
        print(f"  Reliable discovery: {t['reliable_discovery']}")
        print(f"  Total collapse:     {t['total_collapse']}")
        print(f"  Transition width:   {t['transition_width']}")

        print(f"\n  {data['parameter']:<10} {'rate':>6} {'D':>7} {'ticks':>7}")
        print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*7}")
        for param, rate, d, ticks in data["curve"]:
            t_str = f"{ticks:>7}" if ticks else "     --"
            print(f"  {param:<10.3f} {rate:>5.0%} {d:>7.3f} {t_str}")

    print(f"\n{'=' * 70}")
    print("  CORE FINDING")
    print(f"{'=' * 70}")
    print("""
  For a fixed agent class, the space of dynamical systems partitions
  into discoverable and undiscoverable regions separated by narrow
  boundaries (width 0.015-0.070 in the tested parameter).

  - Simple linear systems are most robust to noise.
  - Hidden periodic structure reduces noise tolerance.
  - The logistic map's order-to-chaos transition produces a matching
    discoverability transition, with periodic windows (r=3.63, r=3.74)
    appearing as islands of discoverability within the chaotic regime.

  All claims restricted to the specific agent class and universes tested.
""")

    print(f"{'=' * 70}")
    print("  DISCOVERABILITY INDEX D")
    print(f"{'=' * 70}")
    print("""
  D = 0.50 * accuracy + 0.25 * speed + 0.25 * stability

  accuracy  = max(0, 1 - final_error / 0.05)
  speed     = max(0, 1 - (discovery_tick - 40) / 460)
  stability = max(0, 1 - CV_error / 2)

  D ~ 1: fast, accurate, stable discovery
  D ~ 0: no discovery possible
""")


if __name__ == "__main__":
    print_results()
