# Main Figure Legends

## Figure 1 Study design and fixed comparison

**a**, The public workbook contained 980 numbered candidates. A bounded data-readiness policy admitted 603 A and 119 B records and excluded 258 C records. The 722-record cohort included 536 RAC observations and 186 natural-aggregate controls. Source/recipe grouping produced 563 development observations in 50 groups and 159 holdout observations in 13 groups. Source, group and exact-recipe overlap across the split were zero. **b**, Three dose-based representations were compared using development-only model selection. Thirty matched outer models supported the explanation analyses. Final XGBoost parameters and feature orders were locked before full-development refitting and one holdout prediction batch. The 990 trials refer to all previously completed main and sensitivity searches, not 990 final-test evaluations. Boxes represent actual cohort counts and analytical steps, not simulated observations.

## Figure 2 Prediction depends on validation task and remains limited on the final holdout

**a**, Pooled development OOF RMSE and R² for the identical Physics/histogram-gradient-boosting baseline under random and grouped five-fold validation (n=563). **b**, Mean ± sample SD of five nested grouped XGBoost fold RMSE values for each representation. **c**, Pooled holdout RMSE for all three locked XGBoost models and the development-mean Dummy (n=159, 13 source-disjoint groups). Whiskers are percentile 95% intervals from 2000 paired validation-group bootstrap draws. **d**, The three locked holdout representation contrasts using those same paired draws; negative values favor the first representation under RMSE. All difference intervals include zero. Panels b and c use different variability definitions and must not be read as directly comparable confidence intervals. The holdout is a fixed subset of the same compilation, not newly collected external data. All models and all holdout sources are reported; no model selection followed evaluation.

## Figure 3 Native coordinate explanations depend on representation

**a–c**, Native exact XGBoost TreeSHAP importance for Raw, Physics and Hybrid. Each outer fold first averages absolute contributions within validation sources, then weights sources equally. Bars are means over five folds; whiskers are sample SD. The eight highest fold-mean features are displayed, while all 10/10/20 coordinates remain in supplementary tables. **d–f**, Every development record's native contribution from the model that excluded its source (n=563 per displayed feature). Density stacking changes display positions only. Color is normalized separately by feature to its 5th–95th value percentiles and saturated outside that range; no feature values or SHAP coordinates are removed. References are fold-specific training-path expectations. The panels do not share one common baseline or common player definitions, so unlike variables cannot be compared as causal material effects.

## Figure 4 Common original-material inputs produce a consistent leading block

**a**, Absolute Shapley attribution for a three-player empirical marginal game. Binder contains five original cementitious doses, Water one dose, and Aggregate four aggregate doses. All eight coalitions use the same 64 outer-training background records per fold across representations; all derived values are recomputed after block replacement. Points are means of source-weighted absolute values over five XGBoost folds and whiskers are fold SD, using the original background configuration. Two additional training-background configurations are reported completely in S24–S26; all retained Binder as the leading block. Calculation is exact for this finite game, not the unknown population reference. **b**, Within-source permutation of the same original material blocks, followed by complete representation recomputation. Each fold importance averages eight repeat RMSE increases; points and whiskers summarize the five fold means and SDs. Positive values denote increased prediction error. The Binder block ranked first in all matched folds and both model families; its larger block dimension and sampled variability prevent a per-unit or causal ranking. CatBoost results and global permutation controls are retained in supplementary tables.

## Figure 5 Explanation repeatability does not establish prediction reliability

**a–c**, Mean native feature rank ± sample SD over five XGBoost source-exclusive outer folds. Every coordinate is shown. Sources are equally weighted within folds; rank ties are averaged and Top-k boundaries use fractional credit. Different feature counts imply different rank spaces. **d**, Spearman correlations between all ten pairs of fold importance vectors. **e**, Fractional overlap of the Top-3 sets for the same pairs. Circles indicate XGBoost, squares CatBoost. The ten pairs share only five overlapping training sets and are not independent replications. Strong full-list agreement can coexist with lower leading-set agreement and the weak holdout performance in Fig. 2. Plot jitter changes only categorical display positions.

# Supplementary Figure Legends

## Figure S1 Source heterogeneity and development residuals

Reused from the existing source analysis. All 54 development sources are ordered by mean CS. Panels display all CS observations, four source-mean physical descriptors standardized for descriptive visualization only, Physics/XGBoost OOF residuals, and the prespecified Wasserstein-shift association across 50 validation groups. Source codes map to the supplied source table. Bootstrap intervals are conditional on saved OOF models. Source standardization was not used to fit predictors. Marker displacement is categorical jitter only; no observation is removed.

## Figure S2 Predetermined native dependence views

All 563 XGBoost development OOF records are shown for Raw cement/water, Physics binder mass/dose-derived water-to-binder ratio and Hybrid cement/dose-derived water-to-binder ratio. Color denotes RCA mass fraction. Black lines connect pooled median contributions in up to eight empirical quantile bins with duplicated boundaries merged. Dose units are kg/m³ and SHAP units MPa. These are model dependence summaries with fold-specific references, not experimental dose-response curves or interaction tests.

## Figure S3 All planned dose-shift associations

All three training-relative Raw-dose shift metrics are compared with all three XGBoost representation errors on 50 validation groups. Panels report Spearman correlation and BH-adjusted q from 2000 within-fold permutations. Bootstrap intervals and definitions are supplied in S08. Weak, null and negative relationships are retained. Failure to detect these relationships does not establish the absence of unobserved domain shift.

## Figure S4 CatBoost native explanation sensitivity

CatBoost sensitivity analysis corresponding to main Figure 3. The native coordinate analysis of Fig. 3 is repeated using the same development folds and representations with saved CatBoost outer parameters and exact native contributions. Weighting, full-row inclusion, eight-feature display rule and fold-specific-reference caveats match Fig. 3. It is a sensitivity comparison between learned functions, not a fresh performance search.

## Figure S5 Previously completed repeated-partition results

The original-partition Physics and Hybrid values also appear in the development comparison; they are repeated context, not independent evidence. Pooled nested development OOF RMSE under the original and two alternative source-group partitions, taken unchanged from Stage 2. Physics and Hybrid candidates are XGBoost; Raw's candidate is CatBoost, as identified in the original table. Folds, learner seeds and optimization sampling changed jointly, and the same development observations were reused. These are conditional workflow sensitivities, not independent studies or an isolated feature effect. The original PNG is retained without implying a new vector source.

# Source data and export notes

Main figures are 183 mm PDF/SVG with editable text and 600 dpi PNG previews. Figures 1, 2, 4 and 5 assemble frozen tables; Fig. 3 reuses the Stage 3 native plot. Supplementary figures reuse existing analyses. S5 retains its original PNG format. Source paths and SHA256 hashes are indexed in the review package and numerical tables S01–S26. No simulated measurements are present. Error bars represent fold SD or group-bootstrap intervals only where specified. Ordinary coordinate permutation, common-input Shapley and native SHAP are distinct estimands and are not substituted for each other.


The axis label Dose-derived W/B denotes source-reported water dose divided by the sum of cement and four supplementary cementitious doses. It is nominal and is not a harmonized effective water-to-binder ratio. Historical machine-readable variable name reported_water_binder is retained for byte-identical model compatibility.
