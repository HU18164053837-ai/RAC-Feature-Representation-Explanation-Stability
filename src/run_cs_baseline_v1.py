"""Reproduce deterministic representations, development EDA and grouped baseline.
Run with bundled Python; dependencies reside in project .deps.
The reserved holdout target is never evaluated by this script.
"""
from pathlib import Path
import os, sys, json, hashlib, importlib.metadata
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone

DATA = ROOT / 'data/frozen/cs_v1'
OUT = ROOT / 'results/modeling/cs_v1'
SEED = 20261023
RAW = ['cement','fly_ash','slag','silica_fume','metakaolin','natural_coarse','recycled_coarse','natural_fine','recycled_fine','water']

def representations(d):
    b = d[['cement','fly_ash','slag','silica_fume','metakaolin']].sum(axis=1)
    c = d.natural_coarse + d.recycled_coarse
    f = d.natural_fine + d.recycled_fine
    p = pd.DataFrame({'binder_mass':b, 'reported_water_binder':d.water/b,
        'aggregate_binder':(c+f)/b, 'fine_fraction':f/(c+f),
        'recycled_coarse_fraction':d.recycled_coarse/c,
        'recycled_fine_fraction':d.recycled_fine/f})
    for col in ['fly_ash','slag','silica_fume','metakaolin']:
        p[col+'_binder_fraction'] = d[col]/b
    return {'Raw':d[RAW], 'Physics':p, 'Hybrid':pd.concat([d[RAW],p],axis=1)}

def metrics(y,p):
    return dict(r2=float(r2_score(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),mae=float(mean_absolute_error(y,p)))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=DATA/'modeling.csv'
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['modeling_sha256']
    d=pd.read_csv(path)
    dev,hold=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(d,groups=d.validation_group))
    assert not set(d.iloc[dev].validation_group)&set(d.iloc[hold].validation_group)
    assignment=d[['record_id','source_id','validation_group']].copy()
    assignment['partition']='holdout';assignment.loc[dev,'partition']='development'
    assignment['development_fold']=0
    train=d.iloc[dev].reset_index(drop=True)
    folds=list(GroupKFold(n_splits=5).split(train,groups=train.validation_group))
    for fold,(a,b) in enumerate(folds,1):
        assert not set(train.iloc[a].source_id)&set(train.iloc[b].source_id)
        assert not set(map(tuple,train.iloc[a][RAW].round(8).values))&set(map(tuple,train.iloc[b][RAW].round(8).values))
        assignment.loc[assignment.record_id.isin(train.iloc[b].record_id),'development_fold']=fold
    assignment.to_csv(OUT/'split_assignments.csv',index=False)
    features=representations(d)
    rows=[]
    for name,x in features.items():
        assert np.isfinite(x.values).all()
        pd.concat([d[['record_id']],x],axis=1).to_csv(OUT/(name.lower()+'_features.csv'),index=False)
        rows.extend(dict(representation=name,field=c,definition={
            'binder_mass':'sum(cement,fly_ash,slag,silica_fume,metakaolin)',
            'reported_water_binder':'water/binder_mass; reported water basis varies by source',
            'aggregate_binder':'sum(all four aggregates)/binder_mass',
            'fine_fraction':'sum(fine aggregates)/sum(all aggregates)',
            'recycled_coarse_fraction':'recycled_coarse/sum(coarse aggregates)',
            'recycled_fine_fraction':'recycled_fine/sum(fine aggregates)',
        }.get(c,c.replace('_binder_fraction','')+'/binder_mass' if c.endswith('_binder_fraction') else 'reported dose, kg/m3')) for c in x.columns)
    pd.DataFrame(rows).to_csv(OUT/'feature_dictionary.csv',index=False)
    train[RAW+['cs_mpa']].describe().to_csv(OUT/'development_descriptive_statistics.csv')
    train[RAW+['cs_mpa']].corr(method='spearman').to_csv(OUT/'development_spearman.csv')
    fig,ax=plt.subplots(figsize=(6,4));ax.hist(train.cs_mpa,bins=25,edgecolor='white');ax.set(xlabel='28-day CS (MPa)',ylabel='Development samples');fig.tight_layout();fig.savefig(OUT/'development_cs_distribution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    physics=representations(train)['Physics']
    for ax,col,label in zip(axes,['reported_water_binder','recycled_coarse_fraction'],['Reported water / binder','Recycled coarse fraction']):
        ax.scatter(physics[col],train.cs_mpa,s=12,alpha=.45);ax.set(xlabel=label,ylabel='28-day CS (MPa)')
    fig.tight_layout();fig.savefig(OUT/'development_composition_cs.png',dpi=180);plt.close(fig)
    models={'Mean':DummyRegressor(), 'Ridge':make_pipeline(StandardScaler(),Ridge(alpha=10)),
        'RandomForest':RandomForestRegressor(n_estimators=150,min_samples_leaf=2,random_state=SEED,n_jobs=1),
        'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,learning_rate=.07,l2_regularization=1,early_stopping=False,random_state=SEED)}
    scores=[];predictions=[];summaries=[]
    for rep,full_x in features.items():
        x=full_x.iloc[dev].reset_index(drop=True)
        for name,model in models.items():
            oof=np.full(len(train),np.nan)
            for fold,(a,b) in enumerate(folds,1):
                fitted=clone(model).fit(x.iloc[a],train.cs_mpa.iloc[a])
                p=fitted.predict(x.iloc[b]);oof[b]=p
                scores.append(dict(representation=rep,model=name,fold=fold,n_train=len(a),n_valid=len(b),**metrics(train.cs_mpa.iloc[b],p)))
            assert np.isfinite(oof).all()
            for stratum,mask in [('all',np.ones(len(train),dtype=bool)),('RAC',train.is_rac.values.astype(bool)),('controls',~train.is_rac.values.astype(bool))]:
                summaries.append(dict(representation=rep,model=name,stratum=stratum,n=int(mask.sum()),**metrics(train.cs_mpa[mask],oof[mask])))
            predictions.extend(dict(record_id=int(r),representation=rep,model=name,observed=float(y),predicted=float(p)) for r,y,p in zip(train.record_id,train.cs_mpa,oof))
            print(rep,name,metrics(train.cs_mpa,oof),flush=True)
    pd.DataFrame(scores).to_csv(OUT/'fold_metrics.csv',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'oof_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'development_oof_predictions.csv',index=False)
    chart=summary.query("stratum == 'all' and model != 'Mean'").pivot(index='model',columns='representation',values='rmse')
    ax=chart.plot.bar(figsize=(8,4));ax.set(ylabel='Grouped development OOF RMSE (MPa)',xlabel='');plt.xticks(rotation=0);plt.tight_layout();plt.savefig(OUT/'baseline_rmse.png',dpi=180);plt.close()
    packages=['numpy','pandas','scipy','scikit-learn','matplotlib','openpyxl','joblib','threadpoolctl']
    (OUT/'requirements.txt').write_text('\n'.join(p+'=='+importlib.metadata.version(p) for p in packages)+'\n')
    result=dict(seed=SEED,development_n=len(dev),holdout_n=len(hold),development_groups=train.validation_group.nunique(),holdout_groups=d.iloc[hold].validation_group.nunique(),holdout_evaluated=False,folds=5,validation='publication/recipe-connected GroupKFold on development only',modeling_sha256=manifest['modeling_sha256'],models={k:str(v) for k,v in models.items()})
    (OUT/'run_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__':main()
