"""Final leakage/reproducibility gate and development-only candidate serialization."""
from stage2_models import *
import re
import joblib

def main():
    d,features,folds=load();dev_ids=set(d.record_id)
    # Recover deterministic inner fold assignments for the first process, which
    # predated assignment-file export. This reconstructs splits, never reruns HPO.
    for path in OUT.glob('hpo_*_best.json'):
        tag=path.stem.removesuffix('_best');dest=OUT/(tag+'_inner_assignments.csv')
        if dest.exists():continue
        match=re.search(r'_outer(\d)$',tag)
        sub=d.iloc[folds[int(match[1])-1][0]].reset_index(drop=True) if match else d
        assignment=sub[['record_id','source_id','validation_group']].copy();assignment['inner_fold']=0
        for k,(a,b) in enumerate(GroupKFold(n_splits=3).split(sub,groups=sub.validation_group),1):assignment.loc[b,'inner_fold']=k
        assignment.to_csv(dest,index=False)
    searches=0
    for path in OUT.glob('*_inner_assignments.csv'):
        a=pd.read_csv(path);assert set(a.record_id)<=dev_ids and a.record_id.is_unique
        assert a.groupby('source_id').inner_fold.nunique().max()==1
        sub=d.merge(a[['record_id','inner_fold']],on='record_id')
        assert sub.groupby(RAW).inner_fold.nunique().max()==1
        assert sub.groupby('validation_group').inner_fold.nunique().max()==1
        searches+=1
    assert searches==99
    # Every saved fold prediction must cover only the stated development rows.
    prediction_files=[]
    for pattern in ['matrix_*.csv','nested_*.csv','repeat_*.csv','Aonly_*.csv']:
        for path in OUT.glob(pattern):
            p=pd.read_csv(path)
            if not {'record_id','observed','predicted'}<=set(p.columns):continue
            assert p.record_id.is_unique and set(p.record_id)<=dev_ids
            assert len(p)==(int((d.category=='A').sum()) if path.name.startswith('Aonly_') else 563)
            truth=d.set_index('record_id').cs_mpa.loc[p.record_id].values
            assert np.allclose(p.observed,truth,rtol=0,atol=1e-10)
            if 'fold' in p:
                joined=d.merge(p[['record_id','fold']],on='record_id')
                assert joined.groupby('source_id').fold.nunique().max()==1
                assert joined.groupby(RAW).fold.nunique().max()==1
            prediction_files.append(path.name)
    # Refit one complete outer-fold model from serialized trial parameters.
    a,b=folds[0];params=json.loads((OUT/'hpo_Hybrid_XGBoost_outer1_best.json').read_text())['params']
    m=model('XGBoost',params);m.fit(features['Hybrid'].iloc[a],d.cs_mpa.iloc[a]);pred=m.predict(features['Hybrid'].iloc[b])
    old=pd.read_csv(OUT/'nested_Hybrid_XGBoost.csv').set_index('record_id').predicted.loc[d.record_id.iloc[b]].values
    delta=float(abs(pred-old).max());assert delta<1e-6
    # Candidate models fitted on development only, for future SHAP work, not final claims.
    candidates=pd.read_csv(OUT/'representation_candidates.csv');modelmeta=[]
    for r in candidates.itertuples():
        params=json.loads((OUT/f'hpo_{r.representation}_{r.model}_development_best.json').read_text())['params']
        x=features[r.representation];m=model(r.model,params);m.fit(x,d.cs_mpa)
        path=OUT/f'development_candidate_{r.representation}_{r.model}.joblib';joblib.dump(m,path)
        modelmeta.append(dict(path=path.name,sha256=digest(path),representation=r.representation,model=r.model,features=list(x.columns),training_record_ids=d.record_id.tolist(),parameters=params,holdout_evaluated=False))
    writejson(OUT/'development_model_metadata.json',modelmeta)
    # Full local package lock supplements direct dependency requirements.
    packages={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions(path=[str(ROOT/'.deps')])}
    (OUT/'requirements_full.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items(),key=lambda t:t[0].lower()))+'\n')
    verification=dict(status='PASS',development_n=563,holdout_n=159,holdout_evaluated=False,inner_search_assignment_checks=searches,prediction_files_checked=len(prediction_files),outer_refit_max_prediction_difference=delta,modeling_sha256=digest(FROZEN/'modeling.csv'),split_sha256=digest(BASE/'split_assignments.csv'),scripts_sha256={p.name:digest(p) for p in [ROOT/'src/stage2_models.py',ROOT/'src/stage2_analysis.py',Path(__file__)]})
    writejson(OUT/'verification.json',verification);print(json.dumps(verification))

if __name__=='__main__':main()
