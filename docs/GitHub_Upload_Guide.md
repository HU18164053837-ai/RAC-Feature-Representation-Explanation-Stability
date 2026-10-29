# GitHub upload guide

This folder is a local release candidate; Codex has not created a repository, pushed a commit or published aRelease.

Before upload, confirm author metadata and code/model license, inspect DATA_LICENSE.md, run the safe verifier and review RELEASE_CHECKLIST.md. The pending code-license notice is not an open-source license.

## Browser upload

Create a repository yourself, choose visibility, then upload the contents of RAC956_GitHub_Ready_v1.2 (not its surroundingZIP). Browser upload limits may make GitHubDesktoporGitmore convenient for hundreds of small result files. Do not upload the validationvenv,privatepapersorworkingproject.

## Git or GitHub Desktop

InGitHubDesktop,useanewlocalrepositoryatthisfolderandpublishonlyafterreview.
Equivalent commands, for you to run after creating the real repository:

    git init
    git add .
    git commit -m "Prepare RAC-956 v1.2 release candidate"
    git branch -M main
    git remote add origin ACTUAL_GITHUB_REPOSITORY_URL
    git push -u origin main

No commands above have been executed byCodex. Replace the symbolicURLwithyouractualdestination;nevercommitcredentials. Currentlargestfileisunder5MB;noGitLFSisnecessaryforthecuratedpayload. For laterlargearchives,useaReleaseassetorlong-termrepositoryratherthancheckingduplicateZIPsintoGit.

## Versioned archive

Afterlicense/authorapproval,optionallytagaversionandcreateaGitHubRelease,thenarchivewithZenodooraninstitutionalservice. GitHubaloneisnotanautomaticallyissuedDOI. InsertonlyrealverifiedURL/DOIfieldsinREADME,CITATIONandmanuscript. Untilthen,submissionreadinessremainsCONDITIONAL_PASS.
