# GitHub-ready release QA — PASS

This is a local candidate, not a published repository. Code/model licensing, author metadata and real repository/archive identifiers remain pending. Technical packaging PASS does not authorize or claim publication.

## Clean-environment verification

A new Python 3.12 virtual environment was created with `python -m venv --without-pip .venv`. From the exported repository root, its interpreter executed:

```powershell
.venv/Scripts/python.exe -I scripts/verify_saved_results.py
```

The actual validation environment is outside the release; the command above is the equivalent portable command. Isolated mode and the standard-library-only checker prevent dependence on project PYTHONPATH or installed scientific packages. The final production stdout is included in the complete submission package as `Clean_Environment_Verification.json`.

The gate checks manifest hashes, the 722/563/159 counts, A=603/B=119/C=258, 536 RAC rows, source/group/recipe separation, the one-batch receipt, final locked features/parameters and 12 stored metric values. Model fits and new predictions are both zero. This verifies saved evidence, not prospective model performance or third-party experimental truth.

## Content and safety

- Source-mapped scientific files are byte-identical; the caption document is explicitly mapped to v1.2.
- Data provenance, CC BY 4.0 attribution and derived-data changes are separate from pending code/model licensing.
- No personal absolute user paths, email addresses, detected token/private-key patterns, .env files, runtime environments or temporary caches are bundled.
- No third-party article full-text PDFs are included. Figure PDFs are retained as scientific assets.
- Every current file is below 5 MB, well below GitHub's 100 MB per-file limit.
- The repository has no .git directory and no remote configuration. Zero GitHub account operations, pushes, releases or uploads were executed.
- Legacy training entry points are marked; default verification cannot launch Optuna or holdout inference.

See `Github_Manifest_SHA256.csv`, `docs/Path_Mapping.csv` and `RELEASE_CHECKLIST.md`. The SHA256 manifest excludes itself. Optional plotting/raw-to-freeze commands are not part of the minimal clean-environment gate; their dependencies and boundary are documented separately.
