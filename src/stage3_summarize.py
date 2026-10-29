"""Summaries for source shift, exact TreeSHAP and explanation stability."""
from stage3_compute import *
from scipy.stats import rankdata,spearmanr,kendalltau
from itertools import combinations

def soft_top(values,k):
    v=np.asarray(values);return np.array([np.clip((k-np.sum(v>z))/np.sum(v==z),0,1) for z in v])

def rank_correlation(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if np.std(a)==0 or np.std(b)==0:return dict(spearman=np.nan,kendall=np.nan)
    return dict(spearman=float(spearmanr(a,b).statistic),kendall=float(kendalltau(a,b).statistic))

def summarize():
    d,features,folds=load();idx=d.set_index('record_id')
    importance=[];permutation=[];base_summary=[];concept_importance=[];long_shap=[];interaction=[]
    for family in ['XGBoost','CatBoost']:
        for rep,x in features.items():
            for k in range(1,6):
                stem=f'{family}_{rep}_fold{k}';a=np.load(CACHE/(stem+'.npz'))
                ids=a['record_id'];phi=a['phi'];names=a['features'].tolist();meta=idx.loc[ids]
                source=meta.source_id.to_numpy()
                absphi=pd.DataFrame(abs(phi),columns=names).assign(source_id=source)
                imp=absphi.groupby('source_id').mean().mean().to_numpy()
                rank=rankdata(-imp,method='average');t3=soft_top(imp,3);t5=soft_top(imp,5)
                for j,c in enumerate(names):
                    importance.append(dict(family=family,representation=rep,fold=k,feature=c,concept=concept(c),mean_abs_shap_source_equal=imp[j],mean_abs_shap_row_equal=abs(phi[:,j]).mean(),rank=rank[j],top3=t3[j],top5=t5[j]))
                    long_shap.extend(dict(family=family,representation=rep,fold=k,record_id=int(no),source_id=s,feature=c,value=float(v),shap=float(sh)) for no,s,v,sh in zip(ids,source,a['values'][:,j],phi[:,j]))
                base_summary.append(dict(family=family,representation=rep,fold=k,expected_value_min=float(a['base'].min()),expected_value_max=float(a['base'].max()),validation_sources=len(set(source))))
                cons=[]
                for con in BLOCKS:
                    j=[i for i,c in enumerate(names) if concept(c)==con]
                    val=abs(phi[:,j].sum(axis=1))
                    im=pd.DataFrame({'source':source,'v':val}).groupby('source').v.mean().mean()
                    cons.append((con,float(im),float(np.mean(np.sum(abs(phi[:,j]),axis=1)))))
                vals=[c[1] for c in cons];ranks=rankdata(-np.array(vals));t1=soft_top(vals,1)
                for j,(con,im,mass) in enumerate(cons):concept_importance.append(dict(family=family,representation=rep,fold=k,concept=con,signed_sum_abs_shap_source_equal=im,absolute_feature_mass_row_equal=mass,rank=ranks[j],top1=t1[j]))
                p=pd.read_csv(CACHE/(stem+'_permutation.csv'));p['family']=family;p['representation']=rep;p['fold']=k;permutation.append(p)
                if family=='XGBoost':
                    p=pd.read_csv(CACHE/(stem+'_interaction_pair.csv'));p['representation']=rep;p['fold']=k;interaction.append(p)
    imp=pd.DataFrame(importance);imp.to_csv(OUT/'fold_feature_importance.csv',index=False)
    pd.DataFrame(base_summary).to_csv(OUT/'shap_reference_values.csv',index=False)
    long=pd.DataFrame(long_shap);long.to_csv(OUT/'cross_fitted_shap_long.csv',index=False)
    # Explicit conventional pooled global importance and equal-source global importance.
    global_rows=[]
    for (family,rep,c),g in long.groupby(['family','representation','feature']):
        global_rows.append(dict(family=family,representation=rep,feature=c,pooled_mean_abs_shap=abs(g.shap).mean(),source_equal_mean_abs_shap=g.assign(v=abs(g.shap)).groupby('source_id').v.mean().mean()))
    pd.DataFrame(global_rows).to_csv(OUT/'global_shap_importance.csv',index=False)
    stability=imp.groupby(['family','representation','feature','concept']).agg(mean_rank=('rank','mean'),rank_sd=('rank','std'),top3_frequency=('top3','mean'),top5_frequency=('top5','mean'),mean_importance=('mean_abs_shap_source_equal','mean'),importance_sd=('mean_abs_shap_source_equal','std')).reset_index()
    stability.to_csv(OUT/'feature_rank_stability.csv',index=False)
    cons=pd.DataFrame(concept_importance);cons.to_csv(OUT/'fold_concept_shap.csv',index=False)
    cons.groupby(['family','representation','concept']).agg(mean_rank=('rank','mean'),rank_sd=('rank','std'),top1_frequency=('top1','mean'),mean_importance=('signed_sum_abs_shap_source_equal','mean')).reset_index().to_csv(OUT/'concept_shap_stability.csv',index=False)
    permutation=pd.concat(permutation,ignore_index=True);permutation.to_csv(OUT/'permutation_all_repeats.csv',index=False)
    pf=permutation.groupby(['family','representation','fold','kind','feature']).delta_rmse.agg(['mean','std']).reset_index().rename(columns={'mean':'importance','std':'permutation_repeat_sd'})
    pf['rank']=pf.groupby(['family','representation','fold','kind']).importance.rank(ascending=False,method='average')
    pf['top1']=0.
    for _,g in pf.groupby(['family','representation','fold','kind']):pf.loc[g.index,'top1']=soft_top(g.importance,1)
    pf.to_csv(OUT/'fold_permutation_importance.csv',index=False)
    pf.groupby(['family','representation','kind','feature']).agg(mean_delta_rmse=('importance','mean'),outer_fold_sd=('importance','std'),mean_rank=('rank','mean'),rank_sd=('rank','std'),top1_frequency=('top1','mean')).reset_index().to_csv(OUT/'permutation_rank_stability.csv',index=False)
    # Pairwise fold stability, within feature representation only.
    pairs=[]
    for (family,rep),g in imp.groupby(['family','representation']):
        for a,b in combinations(range(1,6),2):
            q=g.pivot(index='feature',columns='fold',values='mean_abs_shap_source_equal')
            pairs.append(dict(family=family,representation=rep,fold_a=a,fold_b=b,**rank_correlation(q[a],q[b]),top3_fractional_overlap=float(np.minimum(soft_top(q[a],3),soft_top(q[b],3)).sum()/3),top5_fractional_overlap=float(np.minimum(soft_top(q[a],5),soft_top(q[b],5)).sum()/5)))
    pd.DataFrame(pairs).to_csv(OUT/'pairwise_fold_rank_correlations.csv',index=False)
    # Matched-fold between-model concordance. No unlike feature-name correlation.
    between=[]
    for rep in features:
        for k in range(1,6):
            q=imp[(imp.representation==rep)&(imp.fold==k)].pivot(index='feature',columns='family',values='mean_abs_shap_source_equal')
            between.append(dict(representation=rep,fold=k,**rank_correlation(q.XGBoost,q.CatBoost)))
    pd.DataFrame(between).to_csv(OUT/'between_family_feature_concordance.csv',index=False)
    concept_compare=[]
    for kind in ['material_global','material_within_source']:
        for family in ['XGBoost','CatBoost']:
            for k in range(1,6):
                q=pf[(pf.family==family)&(pf.fold==k)&(pf.kind==kind)].pivot(index='feature',columns='representation',values='importance')
                for a,b in combinations(features,2):concept_compare.append(dict(kind=kind,family=family,fold=k,representation_a=a,representation_b=b,**rank_correlation(q[a],q[b])))
    pd.DataFrame(concept_compare).to_csv(OUT/'between_representation_material_concordance.csv',index=False)
    # Coordinate PFI vs native SHAP rank concordance: dependence-sensitive diagnostic.
    concord=[]
    for (family,rep,k),g in imp.groupby(['family','representation','fold']):
        q=g.set_index('feature').mean_abs_shap_source_equal
        p=pf[(pf.family==family)&(pf.representation==rep)&(pf.fold==k)&(pf.kind=='coordinate')].set_index('feature').importance.loc[q.index]
        concord.append(dict(family=family,representation=rep,fold=k,**rank_correlation(q,p)))
    pd.DataFrame(concord).to_csv(OUT/'shap_coordinate_permutation_concordance.csv',index=False)
    pd.concat(interaction,ignore_index=True).to_csv(OUT/'specified_shap_interactions.csv',index=False)
    # Dependency summaries are exploratory, not causal effects; no univariate p-value fishing.
    dependence=[]
    for (family,rep,c),g in long.groupby(['family','representation','feature']):
        rho=rank_correlation(g.value,g.shap)['spearman'];within=[]
        for _,sub in g.groupby('source_id'):
            if len(sub)>=4 and sub.value.nunique()>=3 and sub.shap.nunique()>=3:within.append(rank_correlation(sub.value,sub.shap)['spearman'])
        dependence.append(dict(family=family,representation=rep,feature=c,pooled_value_shap_rho=rho,eligible_sources=len(within),median_within_source_rho=float(np.median(within)) if within else np.nan,positive_within_source_fraction=float(np.mean(np.array(within)>0)) if within else np.nan))
    pd.DataFrame(dependence).to_csv(OUT/'dependence_associations.csv',index=False)
    print('EXPLANATION SUMMARIES COMPLETE',len(long),'feature contributions',flush=True)

def correlations():
    g=pd.read_csv(OUT/'validation_group_shift_errors.csv');n=len(g);rng=np.random.default_rng(SEED)
    draws=rng.integers(0,n,(2000,n));perms=[]
    for _ in range(2000):
        p=np.arange(n)
        for _,sub in g.groupby('fold'):p[sub.index]=rng.permutation(sub.index)
        perms.append(p)
    perms=np.asarray(perms);rows=[]
    for rep in ['Raw','Physics','Hybrid']:
        y=g[rep+'_rmse'].to_numpy()
        for metric in ['wasserstein_shift','mean_standardized_displacement','nearest_distance']:
            x=g[metric].to_numpy();rx=rankdata(x);ry=rankdata(y)
            rho=float(np.corrcoef(rx,ry)[0,1]);zx=(rx-rx.mean())/rx.std();zy=(ry-ry.mean())/ry.std()
            null=np.mean(zx[None,:]*zy[perms],axis=1);p=(1+np.sum(abs(null)>=abs(rho)))/2001
            bx=rankdata(x[draws],axis=1);by=rankdata(y[draws],axis=1);bx-=bx.mean(axis=1,keepdims=True);by-=by.mean(axis=1,keepdims=True)
            boot=np.sum(bx*by,axis=1)/np.sqrt(np.sum(bx**2,axis=1)*np.sum(by**2,axis=1))
            rows.append(dict(representation=rep,shift_metric=metric,n_groups=n,spearman=rho,ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),p_within_fold_permutation=p))
    r=pd.DataFrame(rows);order=np.argsort(r.p_within_fold_permutation.values);q=np.minimum.accumulate((r.p_within_fold_permutation.values[order]*len(r)/np.arange(1,len(r)+1))[::-1])[::-1];r['q_BH']=0.;r.loc[order,'q_BH']=np.clip(q,0,1);r.to_csv(OUT/'shift_error_associations.csv',index=False)
    # Source-level descriptive high/low error profiles; n>=5 for a less noisy split.
    s=pd.read_csv(OUT/'source_distribution_shift_errors.csv');records=[];thresholds=[]
    cols=['cs_mean','RAC_fraction','wasserstein_shift','nearest_distance']+[c+'_mean' for c in RAW]
    for rep in ['Raw','Physics','Hybrid']:
        sub=s[s.n>=5].copy();threshold=sub[rep+'_rmse'].median();sub['error_class']=np.where(sub[rep+'_rmse']>threshold,'higher','lower_or_equal')
        thresholds.append(dict(representation=rep,min_source_n=5,eligible_sources=len(sub),threshold_rmse=threshold))
        for c in cols:
            hi=sub.loc[sub.error_class=='higher',c];lo=sub.loc[sub.error_class=='lower_or_equal',c];sd=np.sqrt((hi.var()+lo.var())/2)
            records.append(dict(representation=rep,feature=c,high_sources=len(hi),low_sources=len(lo),high_mean=hi.mean(),low_mean=lo.mean(),standardized_difference=(hi.mean()-lo.mean())/sd if sd>0 else np.nan))
    pd.DataFrame(records).to_csv(OUT/'higher_lower_error_source_profiles.csv',index=False)
    writejson(OUT/'source_profile_definition.json',thresholds)
    print(r.to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['summarize','correlations']);a=parser.parse_args()
    {'summarize':summarize,'correlations':correlations}[a.phase]()
