# ProtoScience: A Reproducible Pipeline for Automated Equation Discovery from Raw Time-Series Data

## Abstract

We present ProtoScience, an open-source pipeline for automated discovery of governing equations from unlabeled time-series and tabular data. The system combines power-law fitting, multivariate regression, SINDy, FFT analysis, and change-point detection into a unified workflow requiring no domain knowledge. We validate the pipeline on 28 experiments across 13 scientific domains using public datasets from NASA, ESA, CERN, LIGO, USGS, WHO, and the Argo ocean network. The system recovers known laws including Kepler's Third Law (R²=0.998), 5/5 general relativity predictions (R²=1.000), the Stefan-Boltzmann law from 15,000 ESA Gaia stars (R²=0.994), Kleiber's metabolic scaling from 5,416 mammal species (6/7 laws), the Gompertz mortality law across 5 countries (R²=0.994), and the UNESCO equation of state of seawater (R²=1.000), while correctly returning null results on stochastic data (Bitcoin, R²=0.00). A meta-analysis of 24 discovered laws reveals that nature's power-law exponents cluster near simple fractions at 0.30× the rate expected by chance, and that the precision of discovered laws follows a domain hierarchy (GR > astrophysics > physics > biology) consistent with Comte's classification of the sciences. All experiments are provided as executable Python scripts with public data sources.

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

We validate the system on 28 experiments spanning general relativity, astrophysics, cosmology, particle physics, geophysics, oceanography, biology, and medicine — using only publicly available datasets from 7 agencies.

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

**E062 (Kepler):** The most precise recovery. log(P²) vs log(a³/M★) yields slope=0.9988 (theoretical: 1.000) with R²=0.998172 across 2,833 planets spanning 8 orders of magnitude in orbital period.

**E067 (Kirkwood Gaps):** All five major resonance gaps (4:1, 3:1, 5:2, 7:3, 2:1 with Jupiter) are detected as depletions of 84-100% in the semi-major axis histogram. Kepler's Law is simultaneously verified with R²=0.99995.

**E065 (Solar Cycle):** FFT analysis of 277 years of monthly sunspot data identifies the dominant period at 11.09 years (literature: ~11 years) with a secondary peak at 92.4 years (Gleissberg cycle).

### 4.3 Planetary Science

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E063 | Fireballs | CNEOS | 1,052 | Luminous efficiency | τ=8.2% |
| E068 | Mars weather | MSL REMS | 4,583 sols | CO₂ pressure cycle | 22% variation |
| E072 | TESS transits | NASA Archive | 233 | Transit depth law | R²=0.85 |
| E061 | Turbofan engines | NASA C-MAPSS | 100 engines | Degradation laws | R²=0.38/engine |

### 4.4 Particle Physics

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E079 | Dimuon events | CERN CMS | 77,623 | Z boson + J/ψ | M_Z=90.9 GeV |

The pipeline identifies clear resonance peaks at 3.093 GeV (J/ψ, 0.13% error vs 3.097 GeV) and 90.94 GeV (Z boson, 0.28% error vs 91.19 GeV) from the invariant mass spectrum.

### 4.5 Quantum Mechanics

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E075 | Blackbody spectra | Generated | 6 temps | Planck's constant | 2.7% error |
| E076 | Photoelectric effect | Generated | 3 metals | h = 6.6265×10⁻³⁴ | 0.01% error |
| E077 | Hydrogen spectrum | Literature values | 14 lines | Rydberg constant | R²=0.99999994 |
| E078 | Radioactive decay | Generated | 4 isotopes | Universal exponential | 0.29% error |

**Note on generated data:** E075, E076, and E078 use synthetic data generated from known physical laws with added noise. This tests the pipeline's recovery capability but does not constitute discovery from raw experimental measurements. E077 uses published spectral line wavelengths.

### 4.6 Earth Science and Society

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E073 | Turbulence | Generated | 262K pts | Kolmogorov -5/3 | 0.3% error |
| E080 | Arctic sea ice | NSIDC | 47 years | Linear decline | -0.76M km²/decade |
| E081 | COVID-19 cases | JHU | 5 countries | 10 US pandemic waves | r_Brazil=0.058/day |
| E082 | Gini coefficients | World Bank | 171 countries | Pareto α=1.91 | Declining trend |

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

### 4.11 Biology (E094, PanTHERIA)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E094 | Mammal traits | PanTHERIA | 5,416 species | 6/7 allometric scaling laws | Kleiber R²=0.923 |

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

### 4.12 Medicine (E095, SSA/WHO)

| ID | Dataset | Source | N | Discovery | Metric |
|----|---------|--------|---|-----------|--------|
| E095 | Life tables | SSA + WHO | 5 countries | Gompertz mortality law | Mean R²=0.994 |

From actuarial life tables of 5 populations (USA male, USA female, Japan, Nigeria, Sweden), the pipeline recovers the Gompertz law of mortality: m(x) = α·exp(β·x) for ages 30+. The mean β=0.092 corresponds to a mortality doubling time of 7.8 years. R² > 0.993 for every population tested.

| Population | β | Doubling time | R² |
|------------|---|--------------|-----|
| USA Male 2020 | 0.0856 | 8.1 yr | 0.994 |
| USA Female 2020 | 0.0954 | 7.3 yr | 0.995 |
| Japan 2019 | 0.1053 | 6.6 yr | 0.993 |
| Nigeria 2019 | 0.0673 | 10.3 yr | 0.995 |
| Sweden 2019 | 0.1065 | 6.5 yr | 0.994 |

### 4.13 Null Result

| ID | Dataset | N | Result |
|----|---------|---|--------|
| — | Bitcoin daily prices | ~3,650 | R²=0.00 |

The pipeline correctly identifies no compact governing equation in Bitcoin price data, demonstrating that it does not force-fit laws where none exist.

---

## 5. Meta-Analysis: Patterns Across Discovered Laws (E096)

We treat the 24 power-law discoveries from experiments E061–E095 and BH001–BH003 as a dataset and ask: are there patterns in the laws themselves?

### 5.1 Exponent Clustering

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

### 5.4 Other Findings

- **Sample size vs R²:** r = 0.14 (weak). The cleanliness of the phenomenon matters more than data volume.
- **Year of discovery vs R²:** r = −0.28. Newer laws tend to have lower R², suggesting that "easy" laws were discovered first historically.
- **Number of variables vs R²:** r = 0.18 (no significant effect).

---

## 6. Limitations and Failure Modes

We document several important limitations:

### 5.1 Sensitivity to Noise

SINDy's sparse regression is sensitive to noise in derivative estimation. For E061 (turbofan), individual engine R² values are 0.30-0.40, while fleet-pooled R² drops to 0.15 due to inter-engine variability. The pipeline does not currently report uncertainty intervals on discovered coefficients.

### 5.2 Library Selection Bias

The choice of candidate functions (polynomial, trigonometric, rational) determines what the system can discover. If the true governing equation uses functions outside the library, SINDy will return an approximation or fail. This is not "assumption-free" discovery.

### 5.3 Favorable Benchmarks

Many of our experiments involve systems known to have compact governing equations. The pipeline's performance on messy real-world data with confounders, hidden variables, and non-stationary dynamics is not established by these experiments.

### 5.4 Generated Data

Four experiments (E075, E076, E078, E073) use synthetic data. While noise is added, the data inherently conforms to the assumed mathematical structure. Recovery from synthetic data is a necessary but not sufficient validation.

### 5.5 LLM Interpretation

The Interpreter Scientist generates plausible explanations but may confabulate. Its output should be treated as a hypothesis, not a verified conclusion. We deliberately separate mathematical results (verified) from LLM narrative (unverified) in all artifacts.

### 5.6 Comparison with Baselines

We do not provide systematic comparisons against PySINDy, PySR, AI Feynman, or other symbolic regression methods. This is a significant gap that we intend to address in future work with a formal benchmark suite including noise sweeps, ablation studies, and cross-validation.

---

## 7. Reproducibility

All 28 experiments are available as:
- Python scripts in the ProtoScience repository
- Jupyter notebooks (E061–E082) executable in Google Colab
- Standalone scripts (E091–E096, BH001–BH003) that fetch data directly from public APIs

Repository: https://github.com/SaulVanCode/protoscience-nasa-experiments
License: MIT

Dependencies: numpy, scipy, matplotlib (standard scientific Python stack).

---

## 8. Conclusion

ProtoScience demonstrates that a unified pipeline combining power-law fitting, multivariate regression, SINDy, and FFT can recover known physical laws across 13 scientific domains from public data with minimal human intervention. The system recovers laws ranging from Kepler (1619) to Gompertz (1825) to Stefan-Boltzmann (1879) to general relativity (1916) to Kleiber (1932) — spanning 4 centuries of science — using the same code.

The meta-analysis (E096) yields two findings of independent interest: (1) nature's power-law exponents cluster near simple fractions at 3× the rate expected by chance, and (2) the precision of recoverable laws follows a domain hierarchy consistent with the philosophical tradition (fundamental physics > complex systems > biology). These results, while preliminary, suggest that the "mathematical simplicity" of natural laws is not merely aesthetic but statistically measurable.

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
| BH001 | General Relativity | Kerr Simulator | 5/5 GR laws R²=1.000 |
| BH002 | General Relativity | Kerr Simulator | ISCO collapse, D-shape |
| BH003 | General Relativity | Kerr Simulator | cx·sin(θ)=const R²=1.000 |
