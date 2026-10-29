"""Bounded background sensitivity: cached development models only, no fitting."""
from stage3_compute import RAW,BLOCKS,representations,MODELS,ROOT,SEED,digest,writejson
from pathlib import Path
import pandas as pd,numpy as np,joblib
from math import factorial
O=ROOT/'results/modeling/revision_v1.1';O.mkdir(exist_ok=True)
S3=ROOT/'results/modeling/stage3'

def run():
    # This development-only table avoids opening the 722-row matrix or holdout.
    d=pd.read_csv(S3/'development_residual_source_data.csv')
    assert len(d)==563 and d.record_id.nunique()==563
    records=[];backgrounds=[];audits=[]
    configs=[('additional_1',20261101),('additional_2',20261129)]
    for config,seed in configs:
        for fold in range(1,6):
            manifest=__import__('json').loads((S3/'explanations'/f'XGBoost_Raw_fold{fold}_manifest.json').read_text())
            train=manifest['training_ids'];valid_ids=manifest['validation_ids']
            assert not set(train)&set(valid_ids)
            by=d.set_index('record_id');chosen=np.random.default_rng(seed+fold).choice(train,64,replace=False)
            bg=by.loc[chosen,RAW];valid=by.loc[valid_ids].reset_index();n=len(valid)
            backgrounds.extend(dict(configuration=config,seed=seed,fold_seed=seed+fold,fold=fold,record_id=int(i)) for i in chosen)
            rb=np.tile(bg.to_numpy(),(n,1));rv=np.repeat(valid[RAW].to_numpy(),64,axis=0)
            for family in ['XGBoost','CatBoost']:
                for rep in ['Raw','Physics','Hybrid']:
                    stem=f'{config}_{family}_{rep}_fold{fold}';cache=O/(stem+'.csv')
                    mp=MODELS/f'{family}_{rep}_fold{fold}.joblib'
                    mm=__import__('json').loads((S3/'explanations'/f'{family}_{rep}_fold{fold}_manifest.json').read_text())
                    assert digest(mp)==mm['model_sha256']
                    if cache.exists():
                        q=pd.read_csv(cache);audit=__import__('json').loads((O/(stem+'.json')).read_text())
                    else:
                        model=joblib.load(mp);v=np.zeros((n,8))
                        for mask in range(8):
                            raw=rb.copy()
                            for j,cols in enumerate(BLOCKS.values()):
                                if mask&(1<<j):
                                    ix=[RAW.index(c) for c in cols];raw[:,ix]=rv[:,ix]
                            v[:,mask]=model.predict(representations(pd.DataFrame(raw,columns=RAW))[rep]).reshape(n,64).mean(axis=1)
                        phi=np.zeros((n,3))
                        for j in range(3):
                            for mask in range(8):
                                if not mask&(1<<j):
                                    size=mask.bit_count();phi[:,j]+=factorial(size)*factorial(2-size)/6*(v[:,mask|(1<<j)]-v[:,mask])
                        delta=float(abs(phi.sum(axis=1)-(v[:,7]-v[:,0])).max())
                        assert delta<1e-4,'BLOCKED: CORE_METHOD_ERROR additivity'
                        q=pd.DataFrame([dict(configuration=config,seed=seed,family=family,representation=rep,fold=fold,record_id=int(rid),source_id=source,concept=block,shap=float(phi[i,j])) for i,(rid,source) in enumerate(zip(valid.record_id,valid.source_id)) for j,block in enumerate(BLOCKS)])
                        q.to_csv(cache,index=False);audit=dict(configuration=config,family=family,representation=rep,fold=fold,model_sha256=digest(mp),max_additivity_error=delta)
                        writejson(O/(stem+'.json'),audit)
                    records.append(q);audits.append(audit)
            print(config,'fold',fold,'complete',flush=True)
    baseline=pd.read_csv(S3/'material_block_shapley_values.csv');baseline['configuration']='original';baseline['seed']=SEED
    p=pd.concat([baseline]+records,ignore_index=True)
    f=p.assign(abs_shap=abs(p.shap)).groupby(['configuration','seed','family','representation','fold','concept','source_id']).abs_shap.mean().groupby(['configuration','seed','family','representation','fold','concept']).mean().reset_index(name='importance')
    f['rank']=f.groupby(['configuration','family','representation','fold']).importance.rank(ascending=False,method='average')
    f.to_csv(O/'background_fold_importance.csv',index=False)
    margins=[]
    for key,g in f.groupby(['configuration','family','representation','fold']):
        g=g.sort_values('importance',ascending=False);v=g.set_index('concept').importance
        margins.append(dict(zip(['configuration','family','representation','fold'],key),leading_block=g.iloc[0].concept,top_margin=g.iloc[0].importance-g.iloc[1].importance,Binder_minus_Water=v.Binder-v.Water,Binder_minus_Aggregate=v.Binder-v.Aggregate,Water_minus_Aggregate=v.Water-v.Aggregate))
    pd.DataFrame(margins).to_csv(O/'background_leading_blocks_and_margins.csv',index=False)
    f.assign(top1=(f['rank']==1).astype(float)).groupby(['configuration','family','representation','concept']).agg(mean_abs_shap=('importance','mean'),fold_sd=('importance','std'),top1_frequency=('top1','mean')).reset_index().to_csv(O/'background_summary.csv',index=False)
    orig_bg=pd.read_csv(S3/'material_block_training_backgrounds.csv');orig_bg['configuration']='original';orig_bg['seed']=SEED;orig_bg['fold_seed']=orig_bg.fold+SEED
    pd.concat([orig_bg,pd.DataFrame(backgrounds)]).to_csv(O/'all_training_background_ids.csv',index=False)
    writejson(O/'sensitivity_audit.json',dict(status='complete',development_n=563,new_background_configurations=2,background_n=64,model_fits=0,holdout_accessed=False,model_count=30,configurations=configs,audits=audits,source_code_sha256=digest(Path(__file__)),development_table_sha256=digest(S3/'development_residual_source_data.csv')))
    print('COMPLETE',flush=True)
if __name__=='__main__':run()
