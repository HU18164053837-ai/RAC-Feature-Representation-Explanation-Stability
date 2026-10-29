"""Exact three-player empirical marginal Shapley on Raw-dose material blocks.
Uses cached outer models, 64 TRAINING backgrounds matched across representations.
All derived features are recomputed; this is not causal/conditional attribution.
"""
from stage3_compute import *
from stage3_summarize import soft_top,rank_correlation
from math import factorial
from itertools import combinations

def run():
    d,features,folds=load();records=[];backgrounds=[];audit=[]
    for k,(a,b) in enumerate(folds,1):
        bg_indices=np.random.default_rng(SEED+k).choice(a,size=64,replace=False)
        bg=d.iloc[bg_indices][RAW].reset_index(drop=True);valid=d.iloc[b].reset_index(drop=True)
        backgrounds.extend(dict(fold=k,record_id=int(i)) for i in d.record_id.iloc[bg_indices])
        n=len(valid);raw_background=np.tile(bg.to_numpy(),(n,1));raw_valid=np.repeat(valid[RAW].to_numpy(),len(bg),axis=0)
        for family in ['XGBoost','CatBoost']:
            for rep in features:
                m=joblib.load(MODELS/f'{family}_{rep}_fold{k}.joblib')
                v=np.zeros((n,8))
                for mask in range(8):
                    raw=raw_background.copy()
                    for j,block in enumerate(BLOCKS):
                        if mask & (1<<j):
                            cols=[RAW.index(c) for c in BLOCKS[block]];raw[:,cols]=raw_valid[:,cols]
                    xx=representations(pd.DataFrame(raw,columns=RAW))[rep]
                    v[:,mask]=m.predict(xx).reshape(n,len(bg)).mean(axis=1)
                phi=np.zeros((n,3))
                for j in range(3):
                    for mask in range(8):
                        if mask & (1<<j):continue
                        size=mask.bit_count();weight=factorial(size)*factorial(2-size)/factorial(3)
                        phi[:,j]+=weight*(v[:,mask|(1<<j)]-v[:,mask])
                pred=m.predict(features[rep].iloc[b]);err=float(abs(phi.sum(axis=1)+v[:,0]-pred).max());assert err<1e-4
                for j,block in enumerate(BLOCKS):
                    records.extend(dict(family=family,representation=rep,fold=k,record_id=int(no),source_id=s,concept=block,shap=float(z),reference=float(base)) for no,s,z,base in zip(valid.record_id,valid.source_id,phi[:,j],v[:,0]))
                audit.append(dict(family=family,representation=rep,fold=k,max_additivity_error=err,background_n=64,background_training_ids=d.record_id.iloc[bg_indices].tolist()))
                print('BLOCK SHAP',family,rep,k,'additivity',err,flush=True)
    p=pd.DataFrame(records);p.to_csv(OUT/'material_block_shapley_values.csv',index=False)
    pd.DataFrame(backgrounds).to_csv(OUT/'material_block_training_backgrounds.csv',index=False)
    writejson(OUT/'material_block_shapley_audit.json',audit)
    f=p.assign(abs_shap=abs(p.shap)).groupby(['family','representation','fold','concept','source_id']).abs_shap.mean().groupby(['family','representation','fold','concept']).mean().reset_index(name='importance')
    f['rank']=f.groupby(['family','representation','fold']).importance.rank(ascending=False,method='average');f['top1']=0.
    for _,g in f.groupby(['family','representation','fold']):f.loc[g.index,'top1']=soft_top(g.importance,1)
    f.to_csv(OUT/'fold_material_block_shapley.csv',index=False)
    f.groupby(['family','representation','concept']).agg(mean_rank=('rank','mean'),rank_sd=('rank','std'),top1_frequency=('top1','mean'),mean_abs_shap=('importance','mean'),outer_fold_sd=('importance','std')).reset_index().to_csv(OUT/'material_block_shapley_stability.csv',index=False)
    concord=[]
    for family in ['XGBoost','CatBoost']:
        for k in range(1,6):
            q=f[(f.family==family)&(f.fold==k)].pivot(index='concept',columns='representation',values='importance')
            for a,b in combinations(features,2):concord.append(dict(family=family,fold=k,representation_a=a,representation_b=b,**rank_correlation(q[a],q[b])))
    pd.DataFrame(concord).to_csv(OUT/'material_block_shapley_representation_concordance.csv',index=False)

if __name__=='__main__':run()
