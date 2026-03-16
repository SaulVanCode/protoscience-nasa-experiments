# ProtoScience NASA Experiments

**Can an AI rediscover physics from raw NASA data?**

We fed real NASA datasets into [ProtoScience](https://protoscience.ai) — an automated equation discovery engine — with zero domain knowledge. Here's what it found.

## Results at a Glance

| # | Dataset | Discovery | Metric |
|---|---------|-----------|--------|
| E062 | [Exoplanets](notebooks/e062_exoplanets_kepler.ipynb) | **Kepler's Third Law** (1619) | R²=0.998 |
| E063 | [Fireballs](notebooks/e063_fireballs.ipynb) | Luminous efficiency + power law | τ=8.2%, α=0.72 |
| E065 | [Sunspots](notebooks/e065_sunspots.ipynb) | **11.09-year solar cycle** + Gleissberg | FFT exact |
| E066 | [Gravitational Waves](notebooks/e066_gravitational_waves.ipynb) | **Chirp mass formula** verified | R²=0.998 |
| E067 | [Asteroids](notebooks/e067_asteroids_kirkwood.ipynb) | **5/5 Kirkwood gaps** + Kepler | R²=0.99995 |
| E068 | [Mars Weather](notebooks/e068_mars_weather.ipynb) | Seasonal T cycle + CO₂ pressure | 22% variation |

## Quick Start

```bash
# Clone
git clone https://github.com/SaulVanCode/protoscience-nasa-experiments.git
cd protoscience-nasa-experiments

# Install dependencies
pip install -r requirements.txt

# Download all NASA data (~5 MB)
python scripts/download_data.py

# Open any notebook
jupyter notebook notebooks/
```

Or run directly in Google Colab (no install needed):

| Notebook | Colab Link |
|----------|------------|
| Exoplanets: Kepler's Laws | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaulVanCode/protoscience-nasa-experiments/blob/main/notebooks/e062_exoplanets_kepler.ipynb) |
| Fireballs: Impact Physics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaulVanCode/protoscience-nasa-experiments/blob/main/notebooks/e063_fireballs.ipynb) |
| Sunspots: Solar Cycle | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaulVanCode/protoscience-nasa-experiments/blob/main/notebooks/e065_sunspots.ipynb) |
| Gravitational Waves: Chirp Mass | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaulVanCode/protoscience-nasa-experiments/blob/main/notebooks/e066_gravitational_waves.ipynb) |
| **Asteroids: Kirkwood Gaps** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaulVanCode/protoscience-nasa-experiments/blob/main/notebooks/e067_asteroids_kirkwood.ipynb) |
| Mars Weather: CO₂ Cycle | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaulVanCode/protoscience-nasa-experiments/blob/main/notebooks/e068_mars_weather.ipynb) |

## Highlight: Kirkwood Gaps

We gave the system 10,000 asteroid orbits from NASA's JPL database. Without knowing Jupiter exists, it found **5 empty zones** in the asteroid belt — exactly where Jupiter's gravitational resonances clear material out. These are the [Kirkwood gaps](https://en.wikipedia.org/wiki/Kirkwood_gap), discovered by Daniel Kirkwood in 1857.

The system also verified **Kepler's Third Law** with R²=0.99995 — the most precise recovery across all experiments.

## Data Sources

All data comes from official NASA/public sources:

- **Exoplanets**: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) (TAP API)
- **Fireballs**: [NASA CNEOS](https://cneos.jpl.nasa.gov/fireballs/) (SSD API)
- **Sunspots**: [SILSO/Royal Observatory of Belgium](https://www.sidc.be/SILSO/)
- **Gravitational Waves**: [GWOSC](https://gwosc.org/) (LIGO/Virgo/KAGRA)
- **Asteroids**: [JPL Small-Body Database](https://ssd.jpl.nasa.gov/tools/sbdb_query.html) (SSD API)
- **Mars Weather**: [NASA Mars Science Laboratory](https://mars.nasa.gov/) (REMS API)

## How It Works

ProtoScience uses **Sparse Identification of Nonlinear Dynamics (SINDy)** — a method that fits sparse differential equations to time-series data. Combined with FFT analysis, power-law fitting, and change-point detection, it can discover governing equations from raw numerical data without domain knowledge.

Read more at [protoscience.ai](https://protoscience.ai).

## License

MIT. Data is from public NASA sources.
