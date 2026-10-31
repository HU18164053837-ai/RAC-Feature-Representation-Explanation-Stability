# Maintaining the published GitHub repository

The repository is public at https://github.com/HU18164053837-ai/RAC-Feature-Representation-Explanation-Stability . Initial creation/upload is complete; this is no longer a local folder awaiting its first upload. Original local packaging statements described the 24 October 2026 snapshot and are not the present online state.

## Proposed updates

Clone the actual repository, inspect uncommitted work, fetch main and use a dedicated branch. For example:

```sh
git clone https://github.com/HU18164053837-ai/RAC-Feature-Representation-Explanation-Stability.git
cd RAC-Feature-Representation-Explanation-Stability
git switch -c chore/metadata-update
python scripts/verify_saved_results.py
```

After documented changes, update SHA256 entries for changed covered files, inspect the diff, run the safe gate in a clean checkout, commit and push the branch, then open a PR against main. Do not overwrite another person's work or use a force push. These are maintenance instructions, not a claim these example commands were all executed.

Do not upload local environments, private papers, credentials, ZIP duplicates or unrestricted project folders. The original .gitattributes preserves frozen bytes. No model reruns are needed for metadata changes.

## Archive and authorization

No new tag, GitHub Release or permanent DOI is claimed. A future versioned GitHub Release plus Zenodo/institutional archival is optional subject to author/rights-holder approval and journal requirements. Populate only an actually issued DOI. Rights-holder approval of software/model licensing remains outstanding; the source dataset's CC BY 4.0 license is separate.
