# RAC-956: feature representation and explanation stability

Local GitHub-ready candidate v1.2. **Not uploaded or publicly archived.** Author metadata, code/model license, repository URL and archive DOI require human confirmation.

## Study and fixed scope

Raw, Physics and Hybrid represent the same ten material doses. Native importance depends on these coordinates. Repeatable rankings do not establish reliable prediction for excluded studies. The common original-material Shapley game fixes three players across representations, but its Binder priority remains conditional on a 5:1:4 block definition, sampled backgrounds and marginal replacement semantics; it is not causal evidence.

The frozen cohort has 722 records: 536 RAC and 186 natural-aggregate controls. There are 563 development records and 159 already evaluated, source-disjoint holdout records from the same public compilation. This is not newly collected external validation. Final Raw/Physics/Hybrid R² values are −0.100/0.066/0.046 and RMSE values are 20.065/18.487/18.678 MPa. Every learned model had worse MAE than the development-mean Dummy. All paired representation RMSE intervals included zero. These results remain fixed.

## Shortest safe verification

Python 3.12 is recommended. The default gate uses the standard library only. It imports no model packages, loads no model objects, and performs no fitting, search or new prediction.

Windows PowerShell, from the repository root:

```powershell
python -m venv --without-pip .venv
.venv/Scripts/python.exe -I scripts/verify_saved_results.py
```

Linux:

```bash
python3 -m venv --without-pip .venv
.venv/bin/python -I scripts/verify_saved_results.py
```

Direct invocation also works: `python scripts/verify_saved_results.py`.
The gate checks the SHA256 manifest, cohort counts, source/group/recipe separation, receipt, locked parameters and 12 metric values recomputed from **saved** final predictions. It does not evaluate the holdout again. Production QA and commands are documented in `docs/GitHub_Release_QA.md`.

## Repository structure

| Location | Contents |
|---|---|
| `data/raw/` | Unchanged original workbook, with separate data attribution |
| `data/frozen/cs_v1/` | Frozen matrix, dictionary, classification/processing ledgers and evidence snapshots |
| `results/modeling/cs_v1/` | Fixed split indices and baseline predictions |
| `results/modeling/stage2/` | Development predictions, nested search/trial CSV logs, parameters and robustness results |
| `results/modeling/stage3/` | Thirty cached development outer models, fold/background manifests and native/common-block explanations |
| `results/modeling/stage4/` | Locked final models, the consumed holdout's saved predictions, metrics, bootstrap draws and one-batch receipt |
| `results/modeling/revision_v1.1/` | Two additional background configurations, per-record values, summaries and checks |
| `results/publication_tables/` | Tables S01–S26 |
| `figures/main/`, `figures/supplementary/` | Five main and five supplementary bundles; S5 is PNG only |
| `src/` | Scientific numerical and plotting implementations |
| `scripts/` | Safe verification and isolated reproduction wrappers |
| `legacy/` | Historical training/evaluation entry points; not recommended default commands |
| `docs/` | Methods, captions, source references, release instructions and `Path_Mapping.csv` |

Original scientific relative paths are retained where practical. The mapping records every curated source file. Old Word/PDF drafts, private article full texts, installed runtimes, temporary caches and duplicated trial-JSON copies are excluded. Trial CSV logs and selected-parameter JSON files are retained. No Git LFS is needed for the current payload; every file is below 5 MB.

## Optional scientific reproduction

Use a separate pip-enabled environment for optional numerical work:

```powershell
python -m venv .research_env
.research_env/Scripts/python.exe -m pip install -r requirements_research.txt
.research_env/Scripts/python.exe scripts/reproduce_frozen_data.py
```

The freeze wrapper reconstructs six CSV files in a disposable directory from the attributed workbook and frozen decision snapshots, then compares bytes. It does not overwrite release data or independently authenticate third-party experiments. The v1.1 isolated reproduction and method audits are included in the saved results.

Dependency groups are separate: `requirements_research.txt` for numerical methods, `requirements_plotting.txt` for plots, `requirements_documents.txt` for document tools, and `requirements_historical.txt` for optional historical model/search imports. The original environment snapshot is provenance, not an instruction to install an entire old environment. The safe gate needs none of these packages.

For saved-table figure rendering, install research, plotting and historical groups, then run:

```powershell
.research_env/Scripts/python.exe scripts/regenerate_figures.py
```

This reads existing outputs and writes `generated_figures/`; it does not fit or predict. Full original assets remain authoritative. The wrapper preserves the nominal dose-derived W/B label and all 20 Hybrid ranking coordinates. S5 is copied from its sole original PNG. Historical imports may create empty draft-output directories; no release files are overwritten.

Feature conversion is `src/run_cs_baseline_v1.py:representations`. Native/common-input explanation implementations are in `stage3_compute.py` and `stage3_group_shap.py`. The v1.1 sensitivity program uses cached development models only. These historical execution modules are supplied for audit, not scheduled to rerun. In particular, do not execute historical training mains, repeat 990 Optuna trials or call final-test inference for routine verification. See `docs/Reproduction_Workflows.md` and `legacy/README.md`.

## Attribution and licensing

Original database: Haoyun Fan and ShengKang Zhang, Mendeley Data version 2, DOI [10.17632/wc898ff5pj.2](https://doi.org/10.17632/wc898ff5pj.2), **CC BY 4.0**. `DATA_LICENSE.md` identifies third-party ownership and modifications. Article PDFs and third-party runtime binaries are not redistributed.

Code/model licensing is **pending owner approval**: see `LICENSE_CODE_PENDING.md`. No permissive license has been assigned on the authors' behalf. `CITATION.cff` contains explicit unconfirmed author fields and is a release template, not final citation metadata. Replace placeholders only with verified information.

## Human release steps

Review `RELEASE_CHECKLIST.md` and `docs/GitHub_Upload_Guide.md`. The user controls actual GitHub upload and any versioned permanent archive. Insert only real repository/release URLs and issued DOIs. The current local candidate is technically verified, but it is not publicly available, licensed as open source or READY FOR SUBMISSION.
