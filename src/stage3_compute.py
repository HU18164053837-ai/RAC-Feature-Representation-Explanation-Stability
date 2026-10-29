"""Stage 3 explanation calculations; native TreeSHAP, no new search or test scores."""
from stage2_models import ROOT,OUT as STAGE2,BASE,FROZEN,SEED,RAW,load,model,check,metrics,digest,writejson,representations
import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import DMatrix
from catboost import Pool
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

OUT=ROOT/'results/modeling/stage3'
MODELS=OUT/'outer_models'
CACHE=OUT/'explanations'
for p in [OUT,MODELS,CACHE]:p.mkdir(parents=True,exist_ok=True)
BLOCKS={'Binder':['cement','fly_ash','slag','silica_fume','metakaolin'],
        'Water':['water'], 'Aggregate':['natural_coarse','recycled_coarse','natural_fine','recycled_fine']}
REPEATS=8

def concept(feature):
    if feature in BLOCKS['Binder'] or feature=='binder_mass' or feature.endswith('_binder_fraction'):return 'Binder'
    if feature in ['water','reported_water_binder']:return 'Water'
    return 'Aggregate'

def permutation_indices(d,seed,within=False):
    rng=np.random.default_rng(seed)
    if not within:return rng.permutation(len(d))
    ids=np.arange(len(d))
    for s,sub in d.groupby('source_id',sort=True):
        pos=sub.index.to_numpy();ids[pos]=rng.permutation(pos)
    return ids

def compute():
    d,features,folds=load()
    writejson(OUT/'compute_protocol.json',dict(version=1,models=['XGBoost','CatBoost'],representations=list(features),outer_folds=5,permutation_repeats=REPEATS,shap='native exact training-path TreeSHAP',holdout_evaluated=False,modeling_sha256=digest(FROZEN/'modeling.csv'),split_sha256=digest(BASE/'split_assignments.csv')))
    for family in ['XGBoost','CatBoost']:
        for rep,x in features.items():
            old=pd.read_csv(STAGE2/f'nested_{rep}_{family}.csv').set_index('record_id')
            for k,(a,b) in enumerate(folds,1):
                stem=f'{family}_{rep}_fold{k}';done=CACHE/(stem+'_manifest.json')
                if done.exists():continue
                check(d,a,b)
                params_path=STAGE2/f'hpo_{rep}_{family}_outer{k}_best.json'
                params=json.loads(params_path.read_text())['params']
                m=model(family,params);m.fit(x.iloc[a],d.cs_mpa.iloc[a])
                xv=x.iloc[b].reset_index(drop=True);dv=d.iloc[b].reset_index(drop=True)
                pred=m.predict(xv);original=old.loc[dv.record_id,'predicted'].to_numpy()
                delta=float(abs(pred-original).max());assert delta<1e-6
                joblib.dump(m,MODELS/(stem+'.joblib'))
                if family=='XGBoost':
                    contrib=m.get_booster().predict(DMatrix(xv),pred_contribs=True,approx_contribs=False)
                else:
                    contrib=m.get_feature_importance(Pool(xv),type='ShapValues',shap_calc_type='Exact',thread_count=1)
                phi=contrib[:,:-1];base=contrib[:,-1]
                add=float(abs(phi.sum(axis=1)+base-pred).max());assert add<1e-3
                np.savez_compressed(CACHE/(stem+'.npz'),record_id=dv.record_id.to_numpy(),phi=phi,base=base,prediction=pred,values=xv.to_numpy(),features=xv.columns.to_numpy(dtype=str))
                # Standard feature PFI, plus definition-consistent Raw material-block PFI.
                base_rmse=metrics(dv.cs_mpa,pred)['rmse'];rows=[]
                for feature in xv.columns:
                    for r in range(REPEATS):
                        perm=permutation_indices(dv,SEED+100*k+r)
                        xp=xv.copy();xp[feature]=xv[feature].to_numpy()[perm]
                        rows.append(dict(kind='coordinate',feature=feature,repeat=r,delta_rmse=metrics(dv.cs_mpa,m.predict(xp))['rmse']-base_rmse))
                for block,cols in BLOCKS.items():
                    for within in [False,True]:
                        for r in range(REPEATS):
                            perm=permutation_indices(dv,SEED+100*k+r,within)
                            raw=dv[RAW].copy();raw[cols]=dv[cols].to_numpy()[perm]
                            xp=representations(raw)[rep]
                            rows.append(dict(kind='material_within_source' if within else 'material_global',feature=block,repeat=r,delta_rmse=metrics(dv.cs_mpa,m.predict(xp))['rmse']-base_rmse))
                pd.DataFrame(rows).to_csv(CACHE/(stem+'_permutation.csv'),index=False)
                if family=='XGBoost':
                    interaction=m.get_booster().predict(DMatrix(xv),pred_interactions=True,approx_contribs=False)
                    interaction_error=float(abs(interaction.sum(axis=2)-contrib).max());assert interaction_error<1e-3
                    mean_abs=np.mean(abs(interaction[:,:-1,:-1]),axis=0)
                    pd.DataFrame(mean_abs,index=xv.columns,columns=xv.columns).to_csv(CACHE/(stem+'_interaction_matrix.csv'))
                    pair={'Raw':('cement','water'),'Physics':('binder_mass','reported_water_binder'),'Hybrid':('cement','reported_water_binder')}[rep]
                    i,j=[list(xv.columns).index(c) for c in pair]
                    pd.DataFrame({'record_id':dv.record_id,'feature_a':pair[0],'feature_b':pair[1],'interaction_offdiagonal':interaction[:,i,j]}).to_csv(CACHE/(stem+'_interaction_pair.csv'),index=False)
                else:interaction_error=None
                writejson(done,dict(family=family,representation=rep,fold=k,n_train=len(a),n_valid=len(b),parameter_sha256=digest(params_path),oof_prediction_max_difference=delta,shap_additivity_max_error=add,interaction_additivity_max_error=interaction_error,training_ids=d.record_id.iloc[a].tolist(),validation_ids=dv.record_id.tolist(),model_sha256=digest(MODELS/(stem+'.joblib'))))
                print('EXPLAINED',stem,'additivity',add,'PFI repeats',REPEATS,flush=True)
    print('COMPLETE: 30 outer models, zero new HPO trials',flush=True)

def shift():
    d,features,folds=load();sources=[];groups=[]
    for k,(a,b) in enumerate(folds,1):
        train=d.iloc[a];valid=d.iloc[b].copy()
        scale=train[RAW].std(ddof=1).clip(lower=1.)
        center=train[RAW].mean()
        train_std=(train[RAW]-center)/scale
        valid['nearest_distance']=cdist((valid[RAW]-center)/scale,train_std).min(axis=1)/np.sqrt(len(RAW))
        valid['out_of_range_fraction']=((valid[RAW]<train[RAW].min()) | (valid[RAW]>train[RAW].max())).mean(axis=1)
        for key,rows in [('source_id',sources),('validation_group',groups)]:
            for name,g in valid.groupby(key):
                row={key:name,'fold':k,'n':len(g),'RAC_fraction':g.is_rac.mean(),'cs_mean':g.cs_mpa.mean(),'cs_sd':g.cs_mpa.std(),'cs_min':g.cs_mpa.min(),'cs_max':g.cs_mpa.max(),
                    'mean_standardized_displacement':float(abs((g[RAW].mean()-center)/scale).mean()),
                    'wasserstein_shift':float(np.mean([wasserstein_distance(g[c],train[c])/scale[c] for c in RAW])),
                    'nearest_distance':g.nearest_distance.mean(),'out_of_range_fraction':g.out_of_range_fraction.mean()}
                for c in RAW:row[c+'_mean']=g[c].mean()
                for rep in features:
                    pred=pd.read_csv(STAGE2/f'nested_{rep}_XGBoost.csv').set_index('record_id').predicted.loc[g.record_id].to_numpy()
                    row.update({rep+'_'+m:v for m,v in metrics(g.cs_mpa,pred).items()});row[rep+'_bias']=float(np.mean(pred-g.cs_mpa.to_numpy()))
                rows.append(row)
    pd.DataFrame(sources).to_csv(OUT/'source_distribution_shift_errors.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'validation_group_shift_errors.csv',index=False)
    # All observations and residuals, with source names from existing metadata only.
    residual=d.copy()
    ledger=pd.read_csv(FROZEN/'provenance/row_verification_ledger.csv').set_index('No.')
    residual['reference']=residual.record_id.map(ledger['Ref.'])
    for rep in features:
        pred=pd.read_csv(STAGE2/f'nested_{rep}_XGBoost.csv').set_index('record_id').predicted
        residual[rep+'_prediction']=residual.record_id.map(pred);residual[rep+'_residual']=residual[rep+'_prediction']-residual.cs_mpa
    residual.to_csv(OUT/'development_residual_source_data.csv',index=False)
    print('SOURCE DIAGNOSIS',len(sources),'sources',len(groups),'groups',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['explain','shift']);a=parser.parse_args()
    {'explain':compute,'shift':shift}[a.phase]()
