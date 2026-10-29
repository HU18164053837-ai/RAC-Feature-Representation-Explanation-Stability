"""Single authorized batch evaluation of the immutable 159-row holdout.
Does not tune/select models. Subsequent calls only read the completed receipt.
"""
from stage2_models import ROOT,BASE,FROZEN,SEED,RAW,representations,model,digest,writejson
from pathlib import Path
import json,datetime,itertools
import numpy as np
import pandas as pd
import joblib
OUT=ROOT/'results/modeling/stage4';OUT.mkdir(parents=True,exist_ok=True)
REPORT=ROOT/'reports/stage4';REPORT.mkdir(parents=True,exist_ok=True)

def scores(y,p):
    y=np.asarray(y);p=np.asarray(p);den=np.sum((y-y.mean())**2)
    return dict(r2=float(1-np.sum((p-y)**2)/den) if len(y)>1 and den>0 else np.nan,rmse=float(np.sqrt(np.mean((p-y)**2))),mae=float(np.mean(abs(p-y))))

def run():
    receipt=OUT/'test_evaluation_receipt.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['status']=='complete':print('Completed immutable test evaluation already exists; no predictions rerun.');print(json.dumps(r));return
        raise RuntimeError('Existing incomplete receipt requires explicit recovery of the same locked run, not a fresh evaluation.')
    lockfile=ROOT/'results/modeling/stage3/final_analysis_lock.json';lock=json.loads(lockfile.read_text())
    assert digest(FROZEN/'modeling.csv')==lock['modeling_sha256']
    assert digest(BASE/'split_assignments.csv')==lock['split_sha256']
    splits=pd.read_csv(BASE/'split_assignments.csv');full=pd.read_csv(FROZEN/'modeling.csv')
    ids=splits.set_index('record_id').partition
    dev=full[full.record_id.map(ids)=='development'].reset_index(drop=True)
    test=full[full.record_id.map(ids)=='holdout'].reset_index(drop=True)
    assert len(dev)==563 and len(test)==159 and set(dev.record_id)==set(lock['development_ids'])
    overlap=dict(source_overlap=len(set(dev.source_id)&set(test.source_id)),validation_group_overlap=len(set(dev.validation_group)&set(test.validation_group)),rounded_recipe_overlap=len(set(map(tuple,dev[RAW].round(8).values))&set(map(tuple,test[RAW].round(8).values))))
    assert not any(overlap.values())
    xf=representations(dev);xt=representations(test);models={};checks=[]
    models_dir=OUT/'final_models';models_dir.mkdir(exist_ok=True)
    for c in lock['comparators']:
        rep=c['representation'];old_path=ROOT/c['model_file'];assert digest(old_path)==c['model_sha256']
        assert list(xf[rep].columns)==c['features']==list(xt[rep].columns)
        old=joblib.load(old_path);assert all(old.get_params()[k]==v for k,v in c['parameters'].items())
        m=model('XGBoost',c['parameters']);m.fit(xf[rep],dev.cs_mpa)
        difference=float(abs(m.predict(xf[rep])-old.predict(xf[rep])).max());assert difference<1e-6
        models[rep]=m;mp=models_dir/f'{rep}_XGBoost.joblib';joblib.dump(m,mp)
        checks.append(dict(representation=rep,parameters=c['parameters'],features=c['features'],full_development_refit=True,development_prediction_difference_from_locked=difference,model_sha256=digest(mp)))
    protocol=dict(status='locked_before_prediction',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage3_lock_sha256=digest(lockfile),modeling_sha256=digest(FROZEN/'modeling.csv'),split_sha256=digest(BASE/'split_assignments.csv'),development_n=563,test_n=159,development_sources=dev.source_id.nunique(),test_sources=test.source_id.nunique(),development_groups=dev.validation_group.nunique(),test_groups=test.validation_group.nunique(),overlap=overlap,models=checks,selection_rule='all three previously locked XGBoost representations; no post-test selection',preprocessing='deterministic dose-defined features; no scaling, fitted imputation or feature selection',bootstrap_draws=2000,bootstrap_seed=SEED,bootstrap_unit='validation_group',primary_metric='pooled RMSE',secondary_metrics=['R2','MAE','source macro MAE','RAC/control subgroup'],comparisons=[['Physics','Raw'],['Hybrid','Raw'],['Hybrid','Physics']],uncertainty='paired percentile cluster bootstrap; conditional on fitted models and observed test groups; not individual prediction intervals',validation_label='source-disjoint internal holdout from the same compiled database; not a newly collected external dataset')
    writejson(OUT/'pre_evaluation_protocol.json',protocol)
    writejson(receipt,protocol)
    # The single prediction batch begins here, after protocol and model locks.
    predictions=test[['record_id','source_id','validation_group','category','is_rac','cs_mpa']].copy()
    for rep in ['Raw','Physics','Hybrid']:predictions[rep]=models[rep].predict(xt[rep])
    predictions['Dummy']=float(dev.cs_mpa.mean())
    path=OUT/'final_test_predictions.csv';predictions.to_csv(path,index=False)
    groups=sorted(predictions.validation_group.unique());rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(groups),size=(2000,len(groups)))
    grouped=predictions.groupby('validation_group').cs_mpa.agg(n='size',sy='sum',sy2=lambda y:np.sum(y**2)).loc[groups]
    n=grouped.n.values[draws].sum(axis=1);sy=grouped.sy.values[draws].sum(axis=1);sy2=grouped.sy2.values[draws].sum(axis=1);sst=sy2-sy**2/n
    results=[];bootstrap={};sources=[];strata=[]
    for rep in ['Raw','Physics','Hybrid','Dummy']:
        p=predictions[rep];y=predictions.cs_mpa
        tmp=predictions[['validation_group','source_id']].copy();tmp['se']=(p-y)**2;tmp['ae']=abs(p-y)
        pergroup=tmp.groupby('validation_group')[['se','ae']].sum().loc[groups]
        sse=pergroup.se.values[draws].sum(axis=1);sae=pergroup.ae.values[draws].sum(axis=1)
        b=dict(rmse=np.sqrt(sse/n),mae=sae/n,r2=np.where(sst>0,1-sse/sst,np.nan));bootstrap[rep]=b
        r=dict(model=rep,**scores(y,p),source_macro_mae=float(tmp.groupby('source_id').ae.mean().mean()))
        for key,vals in b.items():r[key+'_ci_low']=float(np.nanquantile(vals,.025));r[key+'_ci_high']=float(np.nanquantile(vals,.975))
        results.append(r)
        for s,g in predictions.groupby('source_id'):
            sources.append(dict(model=rep,source_id=s,n=len(g),cs_mean=g.cs_mpa.mean(),bias=float((g[rep]-g.cs_mpa).mean()),**scores(g.cs_mpa,g[rep])))
        for flag,g in predictions.groupby('is_rac'):strata.append(dict(model=rep,stratum='RAC' if flag else 'natural_aggregate_control',n=len(g),sources=g.source_id.nunique(),**scores(g.cs_mpa,g[rep])))
    table=pd.DataFrame(results);table.to_csv(OUT/'final_test_metrics.csv',index=False)
    contrasts=[]
    for a,b in protocol['comparisons']:
        delta=bootstrap[a]['rmse']-bootstrap[b]['rmse']
        contrasts.append(dict(representation_a=a,representation_b=b,rmse_a_minus_b=float(table.set_index('model').loc[a,'rmse']-table.set_index('model').loc[b,'rmse']),ci_low=float(np.quantile(delta,.025)),ci_high=float(np.quantile(delta,.975))))
    pd.DataFrame(contrasts).to_csv(OUT/'final_test_paired_contrasts.csv',index=False)
    pd.DataFrame(sources).to_csv(OUT/'final_test_source_errors.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'final_test_stratum_metrics.csv',index=False)
    np.savez_compressed(OUT/'test_bootstrap_draws.npz',groups=np.array(groups),draws=draws,**{f'{rep}_{metric}':vals for rep,b in bootstrap.items() for metric,vals in b.items()})
    protocol.update(status='complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_evaluated=True,prediction_batches=1,predictions_sha256=digest(path),metrics=results,contrasts=contrasts,test_RAC=int(test.is_rac.sum()),test_controls=int((~test.is_rac).sum()),test_CS_min=float(test.cs_mpa.min()),test_CS_max=float(test.cs_mpa.max()),test_CS_mean=float(test.cs_mpa.mean()),script_sha256=digest(Path(__file__)))
    writejson(receipt,protocol);print(table.to_string(index=False));print(pd.DataFrame(contrasts).to_string(index=False))

if __name__=='__main__':run()
