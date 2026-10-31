# Release QA: historical packaging and current metadata remediation

## Current status (31 October 2026)

The repository is public at https://github.com/HU18164053837-ai/RAC-Feature-Representation-Explanation-Stability . Authors and repository URL are confirmed. No formal software/model license or permanent archive DOI is claimed. Online visibility was read from GitHub repository metadata; it is distinct from the original local packaging acceptance.

The current metadata branch and its clean-checkout execution are documented in Release_Remediation_Report_zh.md. Only documentation, CFF metadata and the corresponding SHA256 inventory may change. The original standard-library verifier is unchanged. It recomputes metrics from saved predictions, never trains, loads a predictive model or generates holdout predictions.

## Historical technical acceptance (24 October 2026)

The exported local candidate passed a fresh no-pip Python 3.12 environment check using `.venv/Scripts/python.exe -I scripts/verify_saved_results.py`. At that historical point it had not been uploaded, had no .git directory and had no repository URL or confirmed author metadata. Those statements are historical, not current. The historical stdout remains in the original submission package as Clean_Environment_Verification.json; this remediation does not rewrite that record.

The historical privacy scan found no personal paths/emails, token/private-key patterns or private article PDFs in that payload. The two author contact emails now added are explicitly supplied and approved for publication by the user. No broad new privacy-scan certification is inferred from the old scan.

## Coverage and interpretation

Github_Manifest_SHA256.csv covers original release files except itself. The initial .gitattributes was added after local export and is included by this remediation. New documentation is also included; no existing entry is removed. Scientific entries retain their original hashes. Tests, all saved results and the safe verifier remain unchanged. Hashes detect byte changes, not the authenticity of third-party experiments or prospective model reliability.

The gate verifies 722/563/159 records, 980 classification rows, A=603/B=119/C=258, 536 RAC rows, source/group/recipe separation, one-batch receipt, locked features/parameters and 12 stored metric values. Optional full numerical reproduction and plotting are not rerun during metadata remediation. No HPO, new split, model fit, new prediction or scientific-result regeneration is performed.
