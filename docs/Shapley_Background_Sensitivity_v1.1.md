# Shapley background sensitivity v1.1

## Decision

PASS for the bounded numerical review. No core-game error was identified. Binder is the leading block in 30/30 models for each of three background configurations. The smallest first-to-second block margin across all configurations/models/folds is 1.148266 MPa. This is descriptive sensitivity of a fixed empirical game, not evidence of causal material dominance.

## Fixed design

Only 563 development records and 30 saved source-exclusive outer models were loaded. Zero fits/HPO and zero holdout access. Original base seed 20261023; two added independent draws use 20261101 and 20261129, each plus fold. Each fold samples 64 training rows without replacement. Original rows are matched across representations and families. Two added configurations meet the specified bounded budget; no third configuration was needed to assess the stated limited question.

Binder=(cement, fly ash, slag, silica fume, metakaolin), Water=(water), Aggregate=(four coarse/fine natural/recycled doses). All eight coalitions recompute the complete representation. For three players, coefficients for coalition sizes 0,1,2 are 1/3,1/6,1/3. Importance first averages absolute Shapley within validation source, then equally across sources. The maximum added-run additivity discrepancy is 7.11e-15 MPa. Every loaded model hash is checked against its historical explanation manifest.

## Magnitudes

Entries are means across five outer folds, in MPa; complete fold values and signed pairwise block margins remain in S24–S26.

| Configuration | Family | Representation | Binder | Water | Aggregate |
|---|---|---|---:|---:|---:|
| additional_1 | CatBoost | Hybrid | 6.591 | 2.920 | 2.943 |
| additional_1 | CatBoost | Physics | 6.574 | 3.219 | 2.978 |
| additional_1 | CatBoost | Raw | 5.304 | 2.085 | 3.092 |
| additional_1 | XGBoost | Hybrid | 6.475 | 3.113 | 3.065 |
| additional_1 | XGBoost | Physics | 7.010 | 3.355 | 3.453 |
| additional_1 | XGBoost | Raw | 5.412 | 2.245 | 3.006 |
| additional_2 | CatBoost | Hybrid | 6.975 | 2.845 | 2.845 |
| additional_2 | CatBoost | Physics | 6.973 | 3.153 | 2.931 |
| additional_2 | CatBoost | Raw | 5.543 | 1.977 | 2.968 |
| additional_2 | XGBoost | Hybrid | 6.832 | 2.997 | 3.015 |
| additional_2 | XGBoost | Physics | 7.414 | 3.235 | 3.443 |
| additional_2 | XGBoost | Raw | 5.664 | 2.104 | 2.833 |
| original | CatBoost | Hybrid | 6.956 | 2.868 | 2.858 |
| original | CatBoost | Physics | 6.941 | 3.148 | 2.949 |
| original | CatBoost | Raw | 5.628 | 2.132 | 2.913 |
| original | XGBoost | Hybrid | 6.823 | 3.055 | 2.981 |
| original | XGBoost | Physics | 7.283 | 3.304 | 3.444 |
| original | XGBoost | Raw | 5.871 | 2.138 | 2.791 |

Water/Aggregate ordering and magnitudes vary. Configurations reuse the same models and validation samples; they are not independent experimental replicates and generate no inferential p-value. The 5:1:4 block dimensions, unequal variation, feature dependence, marginally combined physically infeasible mixtures and source heterogeneity remain unresolved by this check. No sampling-based causal ranking or population-reference coverage is claimed.

## Reproduction

Script: src/revision_v11_sensitivity.py. Outputs: results/modeling/revision_v1.1/background_fold_importance.csv, background_leading_blocks_and_margins.csv, background_summary.csv, all_training_background_ids.csv and sensitivity_audit.json. The original baseline is reused from Stage 3. Individual new per-record values and per-model audits are included. File hashes and seeds are preserved in the reproduction manifest. A separate verification script checks the original-game baseline using a permutation-formula implementation on one existing model/fold.
