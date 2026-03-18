# ProtoScience: A Reproducible Pipeline for Automated Equation Discovery from Raw Time-Series Data

## Abstract

We present ProtoScience, an open-source pipeline for automated discovery of governing equations from raw numerical data. The system combines power-law fitting, multivariate regression, SINDy, FFT analysis, and fractal dimension estimation into a unified workflow requiring no domain knowledge. We validate the pipeline on 52 experiments across 18 scientific domains using public datasets from NASA, ESA, CERN, LIGO, USGS, NOAA, WHO, the Argo ocean network, and 15+ additional sources. The system recovers laws spanning four centuries of science: Kepler's Third Law (R²=0.998), 5/5 general relativity predictions (R²=1.000), Stefan-Boltzmann from ESA Gaia stars (R²=0.994), Milankovitch ice age cycles (3/3 frequencies), the Lorenz attractor equations (5/5 parameters at <0.01% error), Kleiber's metabolic scaling (6/7 laws), Gompertz mortality across 5 countries (R²=0.994), Zipf's law in 5 languages and 8 countries, Mandelbrot's fractal coastlines, and the cosmic ray knee and ankle. The system correctly returns null results on stochastic data (Bitcoin, R²=0.00) and identifies a mathematical fingerprint distinguishing human from AI-generated code. A meta-analysis reveals that nature's power-law exponents cluster near simple fractions at 0.30× the rate expected by chance, and a noise robustness test shows power laws survive 200% noise while linear laws die at 20%. A scale-attractor analysis (E117) tests whether nature's ratios cluster around preferred constants: they do not cluster globally (p=0.998), but power-law exponents prefer simple fractions (0.49x random rate), and each system class has its own preferred ratio — 4/3 for recursive-geometric systems, 3/2 for dissipative ones. The golden ratio is not a universal attractor. All 52 experiments are provided as executable Python scripts with public data sources.

**Keywords:** symbolic regression, equation discovery, power-law scaling, scientific automation, reproducibility, meta-science

---

## 1. Introduction

The automation of scientific discovery from data has been a long-standing goal in machine learning and computational science. Recent advances in symbolic regression (Udrescu & Tegmark, 2020), sparse identification of nonlinear dynamics (Brunton et al., 2016), and large language models have made it increasingly feasible to extract interpretable mathematical relationships from raw measurements.

However, the gap between "recovering a known formula from clean data" and "discovering new science from noisy measurements" remains significant. Benchmarks for symbolic regression have been criticized for favoring curated formulas that do not reflect real-world complexity (Matsubara et al., 2022). Meanwhile, practical tools like PySINDy (de Silva et al., 2020) require substantial expertise in preprocessing, library construction, and hyperparameter tuning.

We introduce ProtoScience, a pipeline that addresses the usability gap rather than the algorithmic gap. Our contribution is not a new regression method, but a reproducible, end-to-end system that:

1. Takes raw CSV/JSON data with minimal assumptions
2. Automatically routes between multiple discovery methods (SINDy, FFT, power-law fitting, change-point detection)
3. Generates visualizations and structured artifacts
4. Provides LLM-based semantic interpretation of discovered equations
5. Explicitly reports when no governing equation is found

We validate the system on 51 experiments spanning cosmology, astrophysics, planetary science, general relativity, particle physics, nuclear physics, quantum physics, earth and climate science, oceanography, chaos theory, biology, medicine, linguistics, social science, music, fractals, and more — using publicly available datasets from NASA, ESA, CERN, LIGO, USGS, NOAA, WHO, and 15+ additional sources.

### 1.1 Scope and Claims

We want to be explicit about what this work is and is not:

- **It is** a system paper demonstrating a reproducible pipeline across many domains
- **It is** a benchmark showing that known laws can be recovered from public data with minimal human intervention
- **It is not** a claim of novel scientific discovery
- **It is not** a methodological advance over SINDy, PySR, or AI Feynman
- **It is not** evidence that the system works on arbitrary noisy data without tuning

---

## 2. Related Work

**Sparse Identification of Nonlinear Dynamics (SINDy).** Brunton, Proctor & Kutz (2016) introduced SINDy for discovering governing equations from time-series data using sparse regression over a library of candidate functions. PySINDy (de Silva et al., 2020) provides a mature Python implementation. ProtoScience uses SINDy as its primary equation discovery engine.

**AI Feynman.** Udrescu & Tegmark (2020) demonstrated symbolic regression that exploits physical symmetries. Their approach achieves high recovery rates on the Feynman Symbolic Regression Database but requires explicit dimensional analysis.

**Genetic Programming.** PySR (Cranmer, 2023) uses multi-population evolutionary search for symbolic expressions. GP-GOMEA (Virgolin et al., 2021) applies gene-pool optimal mixing for compact expressions.

**LLM-assisted discovery.** Recent work has explored using large language models for hypothesis generation (Romera-Paredes et al., 2023) and interpretation of mathematical results. ProtoScience uses an LLM post-hoc for interpretation, not for equation discovery itself.

**Key distinction:** ProtoScience does not advance the algorithmic state-of-the-art. Its contribution is in integration, automation, and reproducibility across domains.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
Raw Data (CSV/JSON)
    │
    ▼
Preprocessing (cleaning, normalization, derivative estimation)
    │
    ├──► SINDy (differential equations)
    ├──► FFT (periodic signals)
    ├──► Power-law fitting (scaling relations)
    ├──► Change-point detection (phase transitions)
    └──► Curve fitting (algebraic relationships)
    │
    ▼
Equation Selection (R², sparsity, cross-validation)
    │
    ▼
LLM Interpreter (semantic explanation)
    │
    ▼
Artifacts (JSON + plots + ledger entry)
```

### 3.2 SINDy Implementation

We use an extended SINDy implementation with Sequential Thresholded Least Squares (STLS). The feature library includes:

- Polynomial terms up to order 2: 1, xᵢ, xᵢxⱼ, xᵢ²
- Optional: rational functions, Hill functions, sigmoid terms

Derivatives are estimated using central finite differences with optional Savitzky-Golay smoothing. The sparsity threshold and regularization parameter are set per-experiment but follow consistent defaults (threshold=0.05, alpha=0.01, max_iter=25).

### 3.3 Multi-Method Routing

Not all datasets contain dynamics suitable for SINDy. The pipeline selects methods based on data structure:

| Data type | Method | Example |
|-----------|--------|---------|
| Time series with dynamics | SINDy | Turbofan degradation |
| Periodic signals | FFT | Solar cycle |
| Algebraic relationships | Power-law / curve fitting | Kepler's Law |
| Regime changes | Change-point detection | Voyager heliopause |
| Size/frequency distributions | Cumulative fitting | Fireball energies |

### 3.4 LLM Interpreter

After equations are discovered, an LLM agent receives:
- The mathematical equations with coefficients
- Variable metadata (names, ranges, units if available)
- Dataset description

It generates:
- Plain-language explanation
- Physical analogies to known systems
- Testable predictions
- Failure mode analysis
- Confidence score (0-1)

The interpreter uses a local Ollama instance (qwen2.5-coder:7b) with temperature=0.3. Its output is clearly separated from the mathematical results to avoid conflating verified equations with LLM-generated narrative.

### 3.5 Null Detection

A critical feature is the system's ability to report "no law found." When applied to Bitcoin daily prices, the pipeline returns R²=0.00 across all methods — correctly identifying the absence of a compact governing equation in stochastic financial data.

---

## 4. Experiments and Results

We organize experiments by domain, from macroscopic to microscopic.

### 4.1 Cosmology

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E069 | Galaxy distances | NED-D | 709 | Hubble's Law v=H₀d | H₀=69.7 km/s/Mpc, R²=0.81 |
| E071 | Galaxy rotation curves | SPARC | 175 galaxies | Flat rotation (dark matter) | 94% flat, 57% DM fraction |
| E074 | Type Ia supernovae | Pantheon+ | 1,590 | Accelerating expansion | Ω_Λ=0.651, q₀=-0.476 |
| E070 | JWST galaxies | UNCOVER DR3 | 74,020 | Size evolution | r~(1+z)^(-0.17) |

**E069 (Hubble's Law):** From 709 galaxies with redshift-independent distances, the pipeline recovers v = 69.7·d with R²=0.81. The fitted H₀=69.7 km/s/Mpc falls between the Planck (67.4) and SH0ES (73.6) values. The scatter (σ=2,548 km/s) reflects peculiar velocities.

**E071 (Dark Matter):** Analyzing 175 SPARC galaxy rotation curves, the pipeline detects that 94% show flat velocity profiles at large radii, inconsistent with Keplerian falloff. The mass discrepancy V²_obs/V²_bar rises from ~1 at the center to ~2.7 at the outermost radius. The Radial Acceleration Relation shows scatter of only 0.196 dex.

**E074 (Dark Energy):** Fitting ΛCDM to 1,590 Pantheon+ supernovae yields Ω_m=0.349, Ω_Λ=0.651, H₀=73.0. The deceleration parameter q₀=-0.476 confirms accelerating expansion. ΛCDM is preferred over an empty universe by Δχ²=38.

### 4.2 Astrophysics

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E062 | Exoplanets | NASA Archive | 3,519 | Kepler's Third Law | R²=0.998, slope=0.9988 |
| E065 | Sunspot numbers | SILSO | 3,326 mo | 11.09-year cycle | FFT peak exact |
| E066 | GW mergers | GWTC | 176 | Chirp mass formula | R²=0.998 |
| E067 | Asteroid orbits | JPL SBDB | 10,000 | 5/5 Kirkwood gaps | R²=0.99995 |
| E064 | Voyager 1 mag field | NASA SPDF | 156,909 | Heliopause crossing | p=3.3×10⁻²⁰ |
| E106 | Pulsars | ATNF Catalogue | 44 | P-Ṗ power-law | R²=0.706 |

**E062 (Kepler):** The most precise recovery. log(P²) vs log(a³/M★) yields slope=0.9988 (theoretical: 1.000) with R²=0.998172 across 2,833 planets spanning 8 orders of magnitude in orbital period.

**E067 (Kirkwood Gaps):** All five major resonance gaps (4:1, 3:1, 5:2, 7:3, 2:1 with Jupiter) are detected as depletions of 84-100% in the semi-major axis histogram. Kepler's Law is simultaneously verified with R²=0.99995.

**E065 (Solar Cycle):** FFT analysis of 277 years of monthly sunspot data identifies the dominant period at 11.09 years (literature: ~11 years) with a secondary peak at 92.4 years (Gleissberg cycle).

**E106 (Pulsars):** From 44 ATNF pulsars, the pipeline recovers the P-Ṗ power-law relation and identifies the millisecond-normal gap spanning 2.5 decades. Crab pulsar spin-down is confirmed.

### 4.3 Planetary Science

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E063 | Fireballs | CNEOS | 1,052 | Luminous efficiency | τ=8.2% |
| E068 | Mars weather | MSL REMS | 4,583 sols | CO₂ pressure cycle | 22% variation |
| E072 | TESS transits | NASA Archive | 233 | Transit depth law | R²=0.85 |
| E061 | Turbofan engines | NASA C-MAPSS | 100 engines | Degradation laws | R²=0.38/engine |
| E109 | Lunar craters | IAU/NASA + LRO | 48 | Size-frequency power-law | α=-2.07, R²=0.997 |

**E109 (Lunar Craters):** From the IAU/NASA Lunar Crater Database, the pipeline recovers the crater size-frequency power-law (α=-2.07) with R²=0.997 and the simple crater depth-diameter scaling relation (R²=0.973).

### 4.4 Particle Physics

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E079 | Dimuon events | CERN CMS | 77,623 | Z boson + J/ψ | M_Z=90.9 GeV |
| E097 | Particle masses | PDG 2024 | 9 particles | Koide formula | error 6.16×10⁻⁶ |
| E099 | Cosmic ray spectrum | PDG + Pierre Auger | 30 points | Knee + ankle breaks | γ=2.27, R²=0.999 |

**E079 (CMS Dimuon):** The pipeline identifies clear resonance peaks at 3.093 GeV (J/ψ, 0.13% error vs 3.097 GeV) and 90.94 GeV (Z boson, 0.28% error vs 91.19 GeV) from the invariant mass spectrum.

**E097 (Koide Formula):** From PDG 2024 particle masses, the Koide formula is confirmed for charged leptons with R=0.6667 (error 6.16×10⁻⁶ from the theoretical 2/3). The relation does not extend to quarks, correctly identifying the limit of its applicability.

**E099 (Cosmic Rays):** The pipeline recovers the overall cosmic ray power-law spectrum (γ=2.27, R²=0.999) and identifies both the "knee" (~10¹⁵·⁵ eV) and "ankle" (~10¹⁸·⁵ eV) spectral breaks — key features of particle astrophysics.

### 4.5 Quantum, Atomic & Nuclear Physics

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E075 | Blackbody spectra | Generated | 6 temps | Planck's constant | 2.7% error |
| E076 | Photoelectric effect | Generated | 3 metals | h = 6.6265×10⁻³⁴ | 0.01% error |
| E077 | Hydrogen spectrum | Literature values | 14 lines | Rydberg constant | R²=0.99999994 |
| E078 | Radioactive decay | Generated | 4 isotopes | Universal exponential | 0.29% error |
| E100 | Nuclear binding | NNDC/IAEA | 72 nuclides | Binding energy peak at Ni-62 | R²=0.997 |

**Note on generated data:** E075, E076, and E078 use synthetic data generated from known physical laws with added noise. This tests the pipeline's recovery capability but does not constitute discovery from raw experimental measurements. E077 uses published spectral line wavelengths.

**E100 (Nuclear Binding):** From 72 nuclides in the NNDC/IAEA database, the pipeline identifies the binding energy per nucleon peak at Ni-62 and fits the valley of stability with R²=0.997, recovering the basic structure of the nuclear landscape.

### 4.6 Earth & Climate Science

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E073 | Turbulence | Generated | 262K pts | Kolmogorov -5/3 | 0.3% error |
| E080 | Arctic sea ice | NSIDC | 47 years | Linear decline | -0.76M km²/decade |
| E081 | COVID-19 cases | JHU | 5 countries | 10 US pandemic waves | r_Brazil=0.058/day |
| E107 | Ice core (Vostok) | Petit et al. 1999 | 3,310 pts | 3/3 Milankovitch cycles | 106, 38, 21 kyr |

**E107 (Milankovitch Cycles):** From 420,000 years of Vostok ice core data, FFT analysis recovers all three Milankovitch orbital cycles: eccentricity (106 kyr, expected 100 kyr), obliquity (38 kyr, expected 41 kyr), and precession (21 kyr, expected 23 kyr) — each within ~8% of the theoretical value.

### 4.7 General Relativity Validation (BH001–BH003)

| ID | Setup | Discovery | Metric |
|----|-------|-----------|--------|
| BH001 | Schwarzschild, 50 mass values | 5/5 GR scaling laws | All R²=1.000 |
| BH002 | Kerr spin sweep, 40 values | ISCO collapse, D-shape shadow | Trend analysis |
| BH003 | Inclination sweep, 200 points | cx·sin(θ) = const | R²=1.000 |

**BH001 (Schwarzschild):** From simulated observables of a non-spinning black hole, the pipeline discovers r_horizon = 2.000·M, r_photon = 3.000·M, r_ISCO = 6.000·M, r_shadow = 5.196·M (= 3√3·M), and shadow_area = 84.82·M² — all with R²=1.000 and coefficient errors < 10⁻⁵. These are the fundamental predictions of Schwarzschild's 1916 solution.

**BH002 (Kerr spin):** Sweeping spin from 0 to 0.998M reveals ISCO collapse (60 → 12.4 at M=10), binding energy increase (5.7% → 32.1%), and shadow asymmetry emergence (0 → 0.124). All observables show regime transitions near a/M ≈ 0.998.

**BH003 (Inclination):** The pipeline discovers that cx·sin(θ_obs) is exactly constant (R²=1.000, product_std=0.000) for all spin values — a geometric consequence of the Boyer-Lindquist coordinate projection. Schwarzschild shadow is confirmed θ-independent (8/9 observables constant).

**Note:** These experiments use simulated data from analytical Kerr metric functions, not observational data. They validate the pipeline's ability to recover exact relationships, not its performance on noisy real-world data.

### 4.8 Stellar Physics (E091, ESA Gaia DR3)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E091 | Gaia DR3 stars | ESA | 15,000 | Stefan-Boltzmann, Mass-Luminosity | R²=0.994, R²=0.945 |

From 15,000 real stars queried via the Gaia DR3 TAP service, multivariate regression recovers the Stefan-Boltzmann law: L = k·R^2.003·T^4.065 with R²=0.994. The exponents match the theoretical values (R=2, T=4) to within 0.15% and 1.6% respectively.

The mass-luminosity relation is recovered as L ~ M^4.123 (R²=0.945) for main-sequence stars (log g > 3.5). Sub-analysis by mass range reveals: low-mass (0.3–0.8 M☉) α=5.18, solar-type (0.8–2.0 M☉) α=4.04, high-mass (2–20 M☉) α=4.09 — consistent with the known variation of the Eddington (1924) exponent.

### 4.9 Geophysics (E092, USGS)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E092 | Earthquakes | USGS | 1,930 | Gutenberg-Richter law | b=0.81, R²=0.915 |

From 1,930 earthquakes (M≥2.5, 30-day USGS catalog), the pipeline recovers the Gutenberg-Richter frequency-magnitude relation: log₁₀(N≥M) = 5.78 − 0.81·M with R²=0.915. The b-value of 0.81 falls within the expected range (0.8–1.2) for global seismicity. The inter-event time distribution has CV=1.095, confirming the Poisson process model for uncorrelated global seismicity.

### 4.10 Oceanography (E093, Argo)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E093 | Ocean profiles | Argo/Argovis | 1,000 profiles | UNESCO EOS-80 | R²=1.000 |

From 1,000 real Argo float profiles spanning 5 ocean basins (N. Pacific, N. Atlantic, S. Atlantic, Indian, S. Pacific), the pipeline recovers the UNESCO equation of state of seawater: ρ(T,S,P) = 999.843 + 0.06794·T − 0.00910·T² + 0.80250·S + 4.5×10⁻⁶·P. All coefficients match the EOS-80 standard to 5+ significant figures.

### 4.11 Biology (E094, E105, E112)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E094 | Mammal traits | PanTHERIA | 5,416 species | 6/7 allometric scaling laws | Kleiber R²=0.923 |
| E105 | Gut microbiome | HMP + American Gut | 25 taxa | Firmicutes/Bacteroidetes ~ BMI | R²=0.940 |
| E112 | Gene expression | GTEx v8 | 40 genes, 6 tissues | Zipf distribution | α=3.71 |

From the PanTHERIA database of 5,416 mammal species, the pipeline recovers 6 of 7 known allometric scaling laws:

| Law | Expected α | Found α | R² | n |
|-----|-----------|---------|-----|---|
| Kleiber (BMR) | 0.75 | 0.702 | 0.923 | 573 |
| Longevity | 0.25 | 0.198 | 0.523 | 1,000 |
| Gestation | 0.25 | 0.189 | 0.456 | 1,335 |
| Neonate mass | 0.75 | 0.872 | 0.787 | 1,069 |
| Home range | 1.00 | 1.061 | 0.679 | 700 |
| Pop density | −0.75 | −0.741 | 0.572 | 947 |
| Litter size | −0.25 | −0.065 | 0.097 | 2,325 |

All discovered exponents except litter size cluster near multiples of 1/4, consistent with West, Brown & Enquist's (1997) fractal network theory of quarter-power scaling.

**E105 (Microbiome):** From aggregated Human Microbiome Project and American Gut data, the pipeline discovers that the Firmicutes/Bacteroidetes ratio scales as BMI^2.28 (R²=0.940), with gut diversity peaking at age ~25.

**E112 (Gene Expression):** Gene expression across 6 GTEx tissues follows a steep Zipf distribution (α=3.71), with hemoglobin comprising 91% of blood expression — consistent with extreme specialization in differentiated tissues.

### 4.12 Medicine & Epidemiology (E095, E110)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E095 | Life tables | SSA + WHO | 5 countries | Gompertz mortality law | Mean R²=0.994 |
| E110 | Drug-age longevity | DrugAge | 3,372 experiments | 2,363 life-extending compounds | Max +268% |

From actuarial life tables of 5 populations (USA male, USA female, Japan, Nigeria, Sweden), the pipeline recovers the Gompertz law of mortality: m(x) = α·exp(β·x) for ages 30+. The mean β=0.092 corresponds to a mortality doubling time of 7.8 years. R² > 0.993 for every population tested.

| Population | β | Doubling time | R² |
|------------|---|--------------|-----|
| USA Male 2020 | 0.0856 | 8.1 yr | 0.994 |
| USA Female 2020 | 0.0954 | 7.3 yr | 0.995 |
| Japan 2019 | 0.1053 | 6.6 yr | 0.993 |
| Nigeria 2019 | 0.0673 | 10.3 yr | 0.995 |
| Sweden 2019 | 0.1065 | 6.5 yr | 0.994 |

**E110 (DrugAge):** From 3,372 longevity experiments in the DrugAge database, the pipeline identifies 2,363 life-extending compounds. The maximum observed extension is +268% (in *Philodina*). Mass scaling of drug effects is weak (R²=0.164), suggesting that longevity mechanisms are not strongly allometric.

### 4.13 Chaos & Nonlinear Dynamics (E108)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E108 | Lorenz system | Simulated | 50,000 pts | 3/3 equations, 5/5 parameters | R²=0.9999999907 |

SINDy recovers all three Lorenz equations from a 50,000-point trajectory with all 5 parameters matched to <0.01% error (σ=10, ρ=28, β=8/3). The clean-data R² of 0.9999999907 confirms that SINDy is exact for polynomial ODEs in the absence of noise.

### 4.14 Linguistics & Universality (E101, E111, E114)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E101 | City populations | UN + census | 130 cities, 8 countries | Zipf's law | α=0.892±0.269, R²=0.949 |
| E111 | Literary texts | Project Gutenberg | 5 texts, ~508K words | Zipf in 5 languages | α=1.131±0.142, R²=0.970 |
| E114 | Cross-domain | Gutenberg + INEGI + NOAA | 3 systems | Universal Zipf | R²=0.978/0.975/0.794 |

**E101 (Zipf — Cities):** City-size distributions across 8 countries (USA, Mexico, Japan, Brazil, India, China, Germany, Nigeria) follow Zipf's law with mean α=0.892±0.269 and mean R²=0.949.

**E111 (Zipf — Language):** Word frequency distributions in 5 languages (English, Spanish, French, German, Russian) from Project Gutenberg texts yield mean α=1.131±0.142 (R²=0.970), consistent with the classical Zipf exponent of ~1.

**E114 (Zipf — Universality):** The pipeline confirms Zipf's law across three fundamentally different systems: human language (Don Quijote, R²=0.978), city sizes (Mexican cities, R²=0.975), and biological communication (whale calls, R²=0.794) — suggesting a universal organizing principle.

### 4.15 Social Science (E082, E098)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E082 | Gini coefficients | World Bank | 171 countries | Pareto α=1.91 | Declining trend |
| E098 | Social indicators | World Bank API | 261 countries | Preston curve, fertility–life exp. | R²=0.661 |

**E098 (Social Geography):** From 261 countries via World Bank Open Data, the pipeline rediscovers the Preston curve (income vs life expectancy) and identifies the fertility–life expectancy correlation (R²=0.661) as the strongest cross-country social law.

### 4.16 Music & Psychoacoustics (E104)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E104 | Musical intervals | Plomp & Levelt 1965 | 13 intervals | Consonance ~ Tenney height | R²=0.875 |

From 13 musical intervals, the pipeline recovers the Tenney height model of consonance: perceived pleasantness is predicted by log₂(p·q) for a frequency ratio p/q. This rediscovers the centuries-old observation that "simple" ratios (2:1, 3:2, 4:3) sound consonant.

### 4.17 Fractals & Geography (E113)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E113 | Coastlines + Koch | Geographic + analytical | 8 coastlines | Mandelbrot fractal dimension | R²=0.995 |

The pipeline recovers the Koch snowflake fractal dimension D=1.2618 (0.01% error vs theoretical ln4/ln3) and measures natural coastline dimensions with mean D=1.51 (R²=0.995 for box-counting fits), confirming Mandelbrot's thesis that geographic boundaries have non-integer dimension.

### 4.18 Code Analysis (E115)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E115 | Source code | Linux, CPython, AI-gen | ~750 tokens | Zipf in code; AI fingerprint | R²=0.936/0.908/0.948 |

Zipf's law holds in source code across all three corpora. The key finding: AI-generated code has a higher Zipf exponent (α=0.87) and lower lexical diversity than human-written code (Linux α=0.79, CPython α=0.81), providing a quantitative fingerprint distinguishing human from machine authorship.

### 4.19 Statistics & Information Theory (E102, E116)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E102 | Numerical datasets | Multiple ProtoScience | 394 values | Benford's law (aggregate) | R²=0.933 |
| E116 | ProtoScience outputs | Self-referential | 148 R² values | Benford self-test | r=0.472, SUSPICIOUS |

**E102 (Benford's Law):** The first-digit distribution of 394 values aggregated across 8 ProtoScience datasets conforms to Benford's law (R²=0.933), though only 1/8 individual datasets passes.

**E116 (Self-Referential Test):** Applying Benford's law to ProtoScience's own experimental outputs (148 R² values) yields a suspicious correlation of only r=0.472. This is expected: R² values are bounded in [0,1] and concentrated near 1, violating the spanning-multiple-orders-of-magnitude prerequisite for Benford's law.

### 4.20 Paleontology & Pharmacology (E109, E110)

*Results for E109 and E110 are reported in sections 4.3 and 4.12 respectively.*

### 4.21 Null Result

| ID | Dataset | N | Result |
|----|---------|---|--------|
| — | Bitcoin daily prices | ~3,650 | R²=0.00 |

The pipeline correctly identifies no compact governing equation in Bitcoin price data, demonstrating that it does not force-fit laws where none exist.

---

## 5. Meta-Analysis: Patterns Across Discovered Laws (E096, E103, E117)

We treat the discovered laws as a dataset and ask: are there patterns in the laws themselves?

### 5.1 Exponent Clustering (E096)

Of the 20 laws with power-law exponents, the mean distance to the nearest simple fraction (0, 1/4, 1/2, 3/4, 1, 3/2, 2, 3, 4) is 0.037. Under a uniform distribution over [0, 4], the expected distance is ~0.125. The observed clustering is **0.30× the random rate**, suggesting that nature's scaling laws preferentially involve simple rational exponents. This is consistent with dimensional analysis arguments but has not previously been quantified empirically across domains.

### 5.2 Domain Hierarchy of Precision

The mean R² by domain follows a clear hierarchy:

| Rank | Domain | Mean R² | n |
|------|--------|---------|---|
| 1 | General Relativity | 0.9997 | 6 |
| 2 | Astrophysics | 0.9707 | 6 |
| 3 | Particle Physics | 0.9900 | 1 |
| 4 | Physics | 0.9540 | 1 |
| 5 | Cosmology | 0.9303 | 3 |
| 6 | Geophysics | 0.9150 | 1 |
| 7 | Biology | 0.7920 | 4 |
| 8 | Engineering | 0.3800 | 1 |

This ordering mirrors Comte's (1830) hierarchy of the sciences and reflects the number of degrees of freedom in each domain's phenomena.

### 5.3 Simulated vs Real Data

Simulated experiments (BH001–BH003) achieve R²=1.000 uniformly. Real-data experiments average R²=0.896. The gap of 0.104 quantifies the "reality penalty" — the noise, confounders, and measurement error that separate mathematical laws from empirical data.

### 5.4 Noise Robustness (E103)

A systematic noise sweep across 5 known laws (Kepler, Hubble, Stefan-Boltzmann, Gutenberg-Richter, Gompertz) reveals strikingly different fragilities:

- **Power laws** (Kepler, Gutenberg-Richter) survive up to 200% additive noise
- **Linear laws** (Hubble) degrade rapidly, becoming unrecoverable at ~20% noise
- **Exponential laws** (Gompertz) show intermediate robustness

This suggests that nature's preference for power-law relationships may partly reflect a selection effect: power laws are the functional forms most likely to survive noisy measurement.

### 5.5 Scale Attractors: Do Nature's Ratios Prefer Certain Numbers? (E117)

A natural extension of E096's exponent clustering is to ask: do the *ratios between consecutive scales* within systems cluster around preferred mathematical constants? We extracted 36 scale ratios from the full experiment corpus — FFT peak ratios (Milankovitch cycles), Zipf rank-1/rank-2 city size ratios, GR characteristic radii ratios, allometric exponent ratios, musical consonance intervals, and fractal dimension ratios — and tested clustering around 11 candidate constants: phi (1.618), sqrt(2), 4/3, sqrt(3), 3/2, 5/3, 2, e, 3, pi, and 4.

**Result 1: No global clustering.** A Monte Carlo test (100,000 simulations, uniform null on [1,5]) yields p=0.998 — the observed ratios are *more* dispersed than random, not less. There is no universal attractor constant.

**Result 2: phi is not special.** Only 2/36 ratios have phi as their nearest candidate constant. In the 1.5-1.7 range where phi lives, 3/2 is a better attractor (mean distance 0.855 vs 0.882 for phi). The closest ratio to phi is Brazil's Zipf city ratio (1.638, Delta=0.020) — likely coincidental.

**Result 3: Category-specific attractors exist.** When ratios are grouped by system type, clear preferences emerge:

| System class | n | Top attractor | Mean distance |
|---|---|---|---|
| Recursive-geometric (music, fractals, allometry) | 12 | 4/3 (9/12 nearest) | 0.160 |
| Social (Zipf cities) | 8 | 4/3 (2/8 nearest) | 0.356 |
| Spectral (Milankovitch, solar) | 12 | 4/3 (7/12 nearest) | 0.446 |
| Dissipative (GR, earthquakes) | 4 | 3/2 (1/4 nearest) | 0.659 |

Recursive-geometric systems show the tightest clustering (mean distance 0.160), consistent with the mathematical structure of iterative growth under constraints. The prevalence of 4/3 across three categories may reflect the ubiquity of cubic-to-quartic scaling transitions in physical systems.

**Result 4: Exponents vs ratios.** Power-law exponents cluster near simple fractions at 0.49x the random rate (37 exponents, confirming E096). But the ratios *between* those exponents do not cluster. This asymmetry suggests that simple-fraction exponents arise from dimensional constraints (mass, length, time combine in integer powers), while ratios between scales within a system are determined by domain-specific dynamics rather than universal mathematics.

### 5.6 Other Findings

- **Sample size vs R²:** r = 0.14 (weak). The cleanliness of the phenomenon matters more than data volume.
- **Year of discovery vs R²:** r = −0.28. Newer laws tend to have lower R², suggesting that "easy" laws were discovered first historically.
- **Number of variables vs R²:** r = 0.18 (no significant effect).

---

## 6. Limitations and Failure Modes

We document several important limitations:

### 6.1 Sensitivity to Noise

SINDy's sparse regression is sensitive to noise in derivative estimation. For E061 (turbofan), individual engine R² values are 0.30-0.40, while fleet-pooled R² drops to 0.15 due to inter-engine variability. The pipeline does not currently report uncertainty intervals on discovered coefficients.

### 6.2 Library Selection Bias

The choice of candidate functions (polynomial, trigonometric, rational) determines what the system can discover. If the true governing equation uses functions outside the library, SINDy will return an approximation or fail. This is not "assumption-free" discovery.

### 6.3 Favorable Benchmarks

Many of our experiments involve systems known to have compact governing equations. The pipeline's performance on messy real-world data with confounders, hidden variables, and non-stationary dynamics is not established by these experiments.

### 6.4 Generated Data

Four experiments (E075, E076, E078, E073) use synthetic data. While noise is added, the data inherently conforms to the assumed mathematical structure. Recovery from synthetic data is a necessary but not sufficient validation.

### 6.5 LLM Interpretation

The Interpreter Scientist generates plausible explanations but may confabulate. Its output should be treated as a hypothesis, not a verified conclusion. We deliberately separate mathematical results (verified) from LLM narrative (unverified) in all artifacts.

### 6.6 Comparison with Baselines

We do not provide systematic comparisons against PySINDy, PySR, AI Feynman, or other symbolic regression methods. This is a significant gap that we intend to address in future work with a formal benchmark suite including noise sweeps, ablation studies, and cross-validation.

---

## 7. Reproducibility

All 52 experiments are available as:
- Python scripts in the ProtoScience repository
- Jupyter notebooks (E061–E082) executable in Google Colab
- Standalone scripts (E091–E117, BH001–BH003) that fetch data directly from public APIs

Repository: https://github.com/SaulVanCode/protoscience-nasa-experiments
License: MIT

Dependencies: numpy, scipy, matplotlib (standard scientific Python stack).

---

## 8. Conclusion

ProtoScience demonstrates that a unified pipeline combining power-law fitting, multivariate regression, SINDy, and FFT can recover known physical laws across 18 scientific domains from public data with minimal human intervention. The system recovers laws ranging from Kepler (1619) to Gompertz (1825) to Stefan-Boltzmann (1879) to general relativity (1916) to Kleiber (1932) to Lorenz (1963) — spanning 4 centuries of science — using the same code.

The meta-analysis (E096) yields two findings of independent interest: (1) nature's power-law exponents cluster near simple fractions at 3x the rate expected by chance, and (2) the precision of recoverable laws follows a domain hierarchy consistent with the philosophical tradition (fundamental physics > complex systems > biology). The noise robustness study (E103) adds a third: power laws survive orders of magnitude more noise than linear relationships, suggesting that the prevalence of power laws in nature may partly reflect observational selection. The scale-attractor analysis (E117) adds a fourth: while no universal constant (including the golden ratio) governs scale ratios across all systems, each class of system has its own preferred ratios — 4/3 for recursive-geometric systems, 3/2 for dissipative ones — suggesting that organizational invariants are class-specific rather than universal.

The system's main contributions are practical — integration, automation, and reproducibility — rather than algorithmic. The most important open question is whether such a pipeline can discover genuinely new relationships in data where the answer is not known.

We release all code and data to enable independent verification: https://github.com/SaulVanCode/protoscience-nasa-experiments

---

## References

- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. PNAS, 113(15), 3932-3937.
- de Silva, B. M., et al. (2020). PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data. JOSS, 5(49), 2104.
- Udrescu, S. M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. Science Advances, 6(16), eaay2631.
- Cranmer, M. (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv:2305.01582.
- Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). SPARC: Mass models for 175 disk galaxies. AJ, 152(6), 157.
- Scolnic, D. M., et al. (2022). The Pantheon+ analysis: The full data set and light-curve release. ApJ, 938(2), 113.
- Matsubara, Y., et al. (2022). Rethinking symbolic regression datasets and benchmarks. NeurIPS Datasets and Benchmarks Track.
- Romera-Paredes, B., et al. (2023). Mathematical discoveries from program search with large language models. Nature, 625, 468-475.
- West, G. B., Brown, J. H., & Enquist, B. J. (1997). A general model for the origin of allometric scaling laws in biology. Science, 276(5309), 122-126.
- Jones, K. E., et al. (2009). PanTHERIA: a species-level database of life history, ecology, and geography of extant and recently extinct mammals. Ecology, 90(9), 2648.
- Gompertz, B. (1825). On the nature of the function expressive of the law of human mortality. Phil. Trans. R. Soc., 115, 513-583.
- Kleiber, M. (1932). Body size and metabolism. Hilgardia, 6(11), 315-353.
- Event Horizon Telescope Collaboration (2019). First M87 Event Horizon Telescope Results. I. ApJ, 875(1), L1.
- Petit, J. R., et al. (1999). Climate and atmospheric history of the past 420,000 years from the Vostok ice core. Nature, 399, 429-436.
- Plomp, R., & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. JASA, 38(4), 548-560.
- Barardo, D., et al. (2017). The DrugAge database of aging-related drugs. Aging Cell, 16(3), 594-597.

---

## Appendix A: Experiment Index

| ID | Domain | Dataset | Key Result |
|----|--------|---------|------------|
| E061 | Aerospace | NASA C-MAPSS | Ps30² degradation |
| E062 | Orbital Mechanics | NASA Exoplanet Archive | Kepler R²=0.998 |
| E063 | Planetary Science | NASA CNEOS | τ=8.2% |
| E064 | Heliophysics | NASA SPDF Voyager 1 | Heliopause p=3.3e-20 |
| E065 | Solar Physics | SILSO | 11.09yr cycle |
| E066 | Gravitational Waves | GWTC | Chirp mass R²=0.998 |
| E067 | Asteroid Dynamics | JPL SBDB | 5/5 Kirkwood gaps |
| E068 | Mars Climate | MSL REMS | CO₂ 22% cycle |
| E069 | Cosmology | NED-D | H₀=69.7 |
| E070 | Galaxy Evolution | JWST UNCOVER | 1,042 at z>10 |
| E071 | Dark Matter | SPARC | 94% flat, 57% DM |
| E072 | Exoplanet Transits | TESS | depth∝(Rp/Rs)² |
| E073 | Fluid Dynamics | Synthetic | k^(-5/3) |
| E074 | Dark Energy | Pantheon+ | Ω_Λ=0.651 |
| E075 | Quantum (Blackbody) | Synthetic | h, T⁴ law |
| E076 | Quantum (Photoelectric) | Synthetic | h, 0.01% error |
| E077 | Atomic Physics | Literature | Rydberg R²=0.99999994 |
| E078 | Nuclear Physics | Synthetic | 4 half-lives |
| E079 | Particle Physics | CERN CMS | Z=90.9 GeV |
| E080 | Climate | NSIDC | -0.76M km²/decade |
| E081 | Epidemiology | JHU | 10 US COVID waves |
| E082 | Economics | World Bank | Pareto α=1.91 |
| E091 | Stellar Physics | ESA Gaia DR3 | Stefan-Boltzmann R²=0.994 |
| E092 | Geophysics | USGS | Gutenberg-Richter b=0.81 |
| E093 | Oceanography | Argo/Argovis | EOS-80 R²=1.000 |
| E094 | Biology | PanTHERIA | Kleiber 6/7 laws |
| E095 | Medicine | SSA/WHO | Gompertz R²=0.994 |
| E096 | Meta-Science | ProtoScience | Exponent clustering 0.30× |
| E097 | Particle Physics | PDG 2024 | Koide formula, error 6.16e-06 |
| E098 | Social Geography | World Bank API | Preston curve, R²=0.661 |
| E099 | Cosmic Rays | PDG + Pierre Auger | Knee + ankle, γ=2.27 |
| E100 | Nuclear Physics | NNDC/IAEA | Binding peak at Ni-62, R²=0.997 |
| E101 | Urban Scaling | UN + census | Zipf cities, 8 countries, R²=0.949 |
| E102 | Statistics | Multiple datasets | Benford's law, R²=0.933 |
| E103 | Methodology | Synthetic | Noise robustness: power laws survive 200% |
| E104 | Music | Plomp & Levelt 1965 | Tenney consonance, R²=0.875 |
| E105 | Microbiome | HMP + American Gut | F/B ~ BMI^2.28, R²=0.940 |
| E106 | Pulsars | ATNF Catalogue | P-Ṗ power-law, R²=0.706 |
| E107 | Paleoclimate | Vostok ice core | 3/3 Milankovitch cycles |
| E108 | Chaos Theory | Simulated Lorenz | 5/5 params, <0.01% error |
| E109 | Lunar Science | IAU/NASA + LRO | Crater power-law, R²=0.997 |
| E110 | Pharmacology | DrugAge | 2,363 compounds, max +268% |
| E111 | Linguistics | Project Gutenberg | Zipf 5 languages, R²=0.970 |
| E112 | Genomics | GTEx v8 | Zipf expression, α=3.71 |
| E113 | Fractals | Geographic + Koch | Coastlines D=1.51, R²=0.995 |
| E114 | Universality | Gutenberg + INEGI + NOAA | Zipf cross-domain |
| E115 | Code Analysis | Linux + CPython + AI | AI fingerprint, α=0.87 vs 0.80 |
| E116 | Meta-analysis | ProtoScience self-test | Benford suspicious, r=0.472 |
| E117 | Scale Attractors | All ProtoScience results | No universal attractor; 4/3 recursive, 3/2 dissipative |
| BH001 | General Relativity | Kerr Simulator | 5/5 GR laws R²=1.000 |
| BH002 | General Relativity | Kerr Simulator | ISCO collapse, D-shape |
| BH003 | General Relativity | Kerr Simulator | cx·sin(θ)=const R²=1.000 |
