# Novelty matrix v1.1

The contribution is an empirical comparison of prediction reliability and attribution comparability under three coordinate representations of the same dose information. It does not introduce SHAP, group Shapley, a new physics-constrained model or the random/group validation gap. Scores across papers are not directly comparable.

| Axis | Miao2024_ref2 | Fan2025_ref3 | Hassan2026_ref4 | Fan2026_ref13 | Present_work |
|---|---|---|---|---|---|
| Database and processing | 980 reported records; multi-response SCM concrete | Four compiled datasets; zero encoding | 520 records / 52 studies | 833 screened mixtures; cylinder conversion | Frozen 722 observations from 980 candidates; no geometry conversion |
| Validation and source independence | Train/validation/test described; source separation not verified from abstract | Benchmark section 3.8 discusses 80:20 and five folds; source-disjoint claim not established | Random versus study-grouped CV is explicit | Fixed random 666/167 split; cross-source limits explicit | Matched source-exclusive development folds; 159-row source-disjoint holdout in same compilation |
| Feature information | SCM inputs; hybrid denotes optimized learners | Material/property descriptors across datasets | Abrams transform, RCA quality index, interactions | Nine inputs including aggregate-quality descriptors | Raw ten doses; Physics invertible; Hybrid redundant; no new measurements |
| Attribution players | Feature-level SHAP | SHAP/PDP examples | Ratio-space SHAP | CatBoost native SHAP | Native coordinates versus identical 5/1/4 Raw-material blocks; derived coordinates recomputed |
| Explanation stability | No absence claim; full text unavailable | Interpretability discussion; matched experiment not established | No absence claim; full text unavailable | Cross-algorithm importance; random-split limitation | Five-fold ranks, dependent pair concordance, Top-3; two extra matched background draws |
| Holdout and engineering boundary | Mix design motivation; no causal conclusion imported | Future AI/engineering framework | Unseen-study degradation already established | Within-database test; demonstration interface | All unfavorable final metrics retained; no optimization or mechanism claim |

## Evidence and access boundaries

- [Miao et al. 2024](https://doi.org/10.1016/j.cscm.2024.e03708): Crossref identity and author-institution abstract verified. Publisher redirect did not provide full text; institutional PDF returned 403. Validation details beyond the abstract remain unspecified.
- [Fan et al. 2025](https://www.techscience.com/CMES/v145n1/64348/html): full publisher text; sections 2.1, 3.8 and 4 support the comparison.
- [Hassan et al. 2026](https://doi.org/10.1016/j.rineng.2026.112641): publisher indexed abstract/highlights and Crossref metadata verified. Direct full-text request returned 403. Crossref assigns volume 32 and December issue while the indexed publisher record lists online availability on 25 August 2026; these dates describe different publication events.
- [Fan et al. 2026](https://doi.org/10.1016/j.isci.2026.117559): indexed publisher full text, especially STAR Methods Dataset splitting/Model interpretation and Limitations, was accessible through search retrieval. Direct-page requests returned 403. Crossref confirms all eight authors, 29(10), 117559.

Raw Crossref JSON records accompany this report. No negative claim about unobserved full-text content is made. Priority cannot be conclusively established from an abstract. The narrower positive description of the present work is retained regardless.
