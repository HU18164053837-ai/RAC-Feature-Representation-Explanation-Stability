"""Independent bounded method check and immutable-result verification."""
from stage2_models import ROOT,RAW,representations,digest
import pandas as pd,numpy as np,json,joblib,itertools
R=ROOT/'reports/revision_v1.1';O=ROOT/'results/modeling/revision_v1.1';S=ROOT/'results/modeling/stage3'
d=pd.read_csv(S/'development_residual_source_data.csv');assert len(d)==563
# Invert the derived representation independently, including all supplementary doses.
p=representations(d)['Physics'];b=p.binder_mass;a=p.aggregate_binder*b;af=p.fine_fraction*a;ac=a-af
inv=pd.DataFrame(index=d.index);inv['water']=p.reported_water_binder*b
for c in ['fly_ash','slag','silica_fume','metakaolin']:inv[c]=p[c+'_binder_fraction']*b
inv['cement']=b-inv[['fly_ash','slag','silica_fume','metakaolin']].sum(axis=1)
inv['recycled_coarse']=p.recycled_coarse_fraction*ac;inv['natural_coarse']=ac-inv.recycled_coarse
inv['recycled_fine']=p.recycled_fine_fraction*af;inv['natural_fine']=af-inv.recycled_fine
inverse_error=float(abs(inv[RAW]-d[RAW]).to_numpy().max());assert inverse_error<1e-9
# Original Raw fold1: direct three-player permutation definition, distinct from
# the subset-weight implementation used by the analysis scripts.
manifest=json.loads((S/'explanations/XGBoost_Raw_fold1_manifest.json').read_text());by=d.set_index('record_id')
bgids=pd.read_csv(S/'material_block_training_backgrounds.csv').query('fold==1').record_id
assert set(bgids)<=set(manifest['training_ids'])
valid=by.loc[manifest['validation_ids']].reset_index();bg=by.loc[bgids,RAW]
n=len(valid);rb=np.tile(bg.to_numpy(),(n,1));rv=np.repeat(valid[RAW].to_numpy(),len(bg),axis=0)
blocks=[['cement','fly_ash','slag','silica_fume','metakaolin'],['water'],['natural_coarse','recycled_coarse','natural_fine','recycled_fine']]
m=joblib.load(S/'outer_models/XGBoost_Raw_fold1.joblib');v={}
for mask in range(8):
    raw=rb.copy()
    for j,cols in enumerate(blocks):
        if mask&(1<<j):
            ix=[RAW.index(c) for c in cols];raw[:,ix]=rv[:,ix]
    v[mask]=m.predict(pd.DataFrame(raw,columns=RAW)).reshape(n,len(bg)).mean(axis=1)
phi=np.zeros((n,3))
for order in itertools.permutations(range(3)):
    mask=0
    for j in order:
        nxt=mask|(1<<j);phi[:,j]+=(v[nxt]-v[mask])/6;mask=nxt
old=pd.read_csv(S/'material_block_shapley_values.csv').query("family=='XGBoost' and representation=='Raw' and fold==1")
old=old.pivot(index='record_id',columns='concept',values='shap').loc[valid.record_id,['Binder','Water','Aggregate']]
original_error=float(abs(phi-old.to_numpy()).max());assert original_error<1e-5,'BLOCKED: CORE_METHOD_ERROR original game mismatch'
bg=pd.read_csv(O/'all_training_background_ids.csv')
for (cfg,fold),g in bg.groupby(['configuration','fold']):
    mm=json.loads((S/'explanations'/f'XGBoost_Raw_fold{fold}_manifest.json').read_text())
    assert len(g)==64 and g.record_id.nunique()==64 and set(g.record_id)<=set(mm['training_ids'])
orig=pd.read_csv(S/'fold_material_block_shapley.csv');new=pd.read_csv(O/'background_fold_importance.csv').query("configuration=='original'")
merged=orig.merge(new,on=['family','representation','fold','concept'],suffixes=('_old','_new'));assert len(merged)==90
assert np.allclose(merged.importance_old,merged.importance_new,rtol=0,atol=1e-10)
receipt=json.loads((ROOT/'results/modeling/stage4/test_evaluation_receipt.json').read_text())
assert receipt['prediction_batches']==1
assert digest(ROOT/'data/frozen/cs_v1/modeling.csv')==receipt['modeling_sha256']
assert digest(ROOT/'results/modeling/cs_v1/split_assignments.csv')==receipt['split_sha256']
oldmanifest=json.loads((ROOT/'results/modeling/stage4/package_verification.json').read_text())['files']
for name,sha in oldmanifest.items():assert digest(ROOT/name)==sha,('v1.0 changed',name)
P=ROOT/'manuscript/v1.1';text=(P/'Manuscript_v1.1.md').read_text(encoding='utf8')
assert 'reported water/binder' not in text.lower() and 'reported water-binder' not in text.lower()
for field in ['−0.100','20.065','13.999','0.066','18.487','13.090','0.046','18.678','12.998','−0.005','19.174','12.357']:
    assert field in text
refs=text.split('## References')[1];assert len(__import__('re').findall(r'^\[\d+\]',refs,flags=__import__('re').M))==13
assert len(list((P/'Figures_Main').glob('*.pdf')))==5
assert len(list((P/'Figures_Supplementary').glob('*.png')))==5
assert len(list((P/'Tables_Supplementary').glob('*.csv')))==26
result=dict(status='PASS',original_game_permutation_check_max_difference=original_error,development_feature_inverse_max_error=inverse_error,original_fold_importance_max_difference=float(abs(merged.importance_old-merged.importance_new).max()),background_membership_checks=15,development_n=563,frozen_matrix_n=722,holdout_n=159,test_predictions_rerun=False,v1_0_package_hashes_unchanged=True,receipt_sha256=digest(ROOT/'results/modeling/stage4/test_evaluation_receipt.json'),metrics_sha256=digest(ROOT/'results/modeling/stage4/final_test_metrics.csv'),source_code_sha256=digest(__import__('pathlib').Path(__file__)))
(O/'independent_method_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
