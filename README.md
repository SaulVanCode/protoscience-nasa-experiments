# ProtoScience: Automated Equation Discovery from Public Data

An open-source pipeline that takes raw time-series data (CSV/JSON) and discovers governing equations using SINDy, FFT, power-law fitting, and change-point detection. Built on [PySINDy](https://pysindy.readthedocs.io/) and standard scientific Python.

**This is not a new algorithm** — it's a reproducible workflow that automates preprocessing, multi-method routing, and interpretation across many domains. All results are rediscoveries of known laws from public datasets.

## Results

| # | Dataset | Source | Discovery | Metric |
|---|---------|--------|-----------|--------|
| E061 | [Turbofan engines](notebooks/e061_turbofan.ipynb) | NASA C-MAPSS | Degradation laws (Ps30²) | R²=0.38/engine |
| E062 | [Exoplanets](notebooks/e062_exoplanets_kepler.ipynb) | NASA Archive | **Kepler's Third Law** | R²=0.998 |
| E063 | [Fireballs](notebooks/e063_fireballs.ipynb) | NASA CNEOS | Luminous efficiency | τ=8.2% |
| E064 | [Voyager 1](notebooks/e064_voyager.ipynb) | NASA SPDF* | Heliopause crossing | p=3.3e-20 |
| E065 | [Sunspots](notebooks/e065_sunspots.ipynb) | SILSO | **11.09-year solar cycle** | FFT exact |
| E066 | [Gravitational Waves](notebooks/e066_gravitational_waves.ipynb) | GWTC | **Chirp mass formula** | R²=0.998 |
| E067 | [Asteroids](notebooks/e067_asteroids_kirkwood.ipynb) | JPL SBDB | **5/5 Kirkwood gaps** | R²=0.99995 |
| E068 | [Mars Weather](notebooks/e068_mars_weather.ipynb) | MSL REMS | CO₂ pressure cycle | 22% variation |
| E069 | [Hubble's Law](notebooks/e069_hubble_law.ipynb) | NED-D | **Universe expanding** | H₀=69.7 |
| E070 | [JWST Galaxies](notebooks/e070_jwst_galaxies.ipynb) | UNCOVER DR3 | Size evolution | 1,042 at z>10 |
| E071 | [Dark Matter](notebooks/e071_dark_matter.ipynb) | SPARC | **Flat rotation curves** | 94% flat, 57% DM |
| E072 | [TESS Transits](notebooks/e072_tess_transits.ipynb) | NASA Archive | Transit depth law | R²=0.85 |
| E074 | [Dark Energy](notebooks/e074_dark_energy.ipynb) | Pantheon+ | **Accelerating expansion** | Ω_Λ=0.651 |
| E079 | [CERN Dimuon](notebooks/e079_cern_dimuon.ipynb) | CERN CMS | **Z boson + J/ψ** | M_Z=90.9 GeV |
| E080 | [Arctic Ice](notebooks/e080_arctic_ice.ipynb) | NSIDC | Linear decline | -0.76M km²/decade |
| E082 | [Inequality](notebooks/e082_pareto_inequality.ipynb) | World Bank | Pareto law | α=1.91 |
| E090 | [Selection by Inferability](notebooks/e090_selection_by_inferability.py) | Simulation | **Phase transitions in discoverability** | width=0.015 |
| — | Bitcoin | — | **No law found** | R²=0.00 |

*E064 uses realistic generated data matching published Voyager 1 characteristics.

## Quick Start

```bash
git clone https://github.com/SaulVanCode/protoscience-nasa-experiments.git
cd protoscience-nasa-experiments
pip install -r requirements.txt
jupyter notebook notebooks/
```

Or run any notebook directly in **Google Colab** (no install needed) — click the Colab badge at the top of each notebook.

## LLM Interpreter

The `interpreter/` directory contains an LLM-based agent that takes discovered equations and generates plain-language explanations, physical analogies, and testable predictions. See [interpreter/README.md](interpreter/README.md) for usage.

## Limitations

- **No methodological novelty** — this is PySINDy + FFT + fitting, well-packaged
- **Only rediscoveries** — no new scientific insights, only recovery of known laws
- **Favorable benchmarks** — datasets chosen because they have known compact equations
- **No formal comparison** against PySINDy, PySR, or AI Feynman baselines
- **No uncertainty quantification** on discovered coefficients
- **LLM interpreter may confabulate** — its output is narrative, not verified math

## Data Sources

All data from official public sources:

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | [NASA CNEOS](https://cneos.jpl.nasa.gov/fireballs/) | [SILSO](https://www.sidc.be/SILSO/) | [GWOSC](https://gwosc.org/) | [JPL SBDB](https://ssd.jpl.nasa.gov/tools/sbdb_query.html) | [MSL REMS](https://mars.nasa.gov/) | [NED-D](https://ned.ipac.caltech.edu/Library/Distances/) | [JWST UNCOVER](https://jwst-uncover.github.io/DR3.html) | [SPARC](https://astroweb.case.edu/SPARC/) | [Pantheon+](https://github.com/PantheonPlusSH0ES/DataRelease) | [CERN Open Data](https://opendata.cern.ch/) | [NSIDC](https://nsidc.org/data/seaice_index) | [World Bank](https://data.worldbank.org/)

## How It Works

The pipeline combines multiple discovery methods:

1. **SINDy** (Brunton et al., 2016) — sparse regression over candidate function libraries for differential equations
2. **FFT** — periodic signal detection
3. **Power-law / curve fitting** — algebraic relationships
4. **Change-point detection** — phase transitions and regime shifts

It does not advance the algorithmic state-of-the-art. The contribution is integration, automation, and reproducibility.

## Paper

A draft paper is in `paper/protoscience_paper.md`. Feedback welcome.

## License

MIT
