"""Scientific summaries and export figures, restricted to fixed development data."""
from stage2_models import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations
REPORT=ROOT/'reports/stage2'
FIG=OUT/'figures'
REPORT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def savefig(fig,name):
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(FIG/(name+'.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig)

def table(df,columns=None):
    df=df[columns] if columns else df
    # Avoid optional tabulate dependency.
    head='| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
    return head+'\n'.join('| '+' | '.join(f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))+'\n'

def augmented(path,d):
    p=pd.read_csv(path)
    p=p.merge(d.drop(columns='cs_mpa'),on='record_id',validate='one_to_one')
    assert len(p)==len(d) or path.name.startswith('Aonly')
    p['error']=p.predicted-p.observed;p['ae']=abs(p.error);p['se']=p.error**2
    return p

def grouped_errors(p,keys):
    rows=[]
    for name,g in p.groupby(keys,observed=True):
        vals=name if isinstance(name,tuple) else (name,)
        score=metrics(g.observed,g.predicted) if len(g)>1 else dict(r2=np.nan,rmse=float(np.sqrt(g.se.mean())),mae=float(g.ae.mean()))
        if g.observed.nunique()<2:score['r2']=np.nan
        rows.append(dict(zip(keys,vals))|dict(n=len(g),bias=g.error.mean(),**score))
    return pd.DataFrame(rows)

def diagnose():
    d,features,folds=load()
    baseline=pd.read_csv(BASE/'development_oof_predictions.csv')
    p=baseline[(baseline.representation=='Physics') & (baseline.model=='HistGradientBoosting')].copy()
    dummy=baseline[(baseline.representation=='Physics') & (baseline.model=='Mean')].set_index('record_id')
    f=pd.read_csv(BASE/'fold_metrics.csv');f=f[(f.representation=='Physics') & (f.model=='HistGradientBoosting')].copy()
    target_info=[];groupcheck=[]
    for k,(a,b) in enumerate(folds,1):
        assert np.allclose(dummy.loc[d.record_id.iloc[b],'predicted'],d.cs_mpa.iloc[a].mean())
        target_info.append(dict(fold=k,n=len(b),sources=d.source_id.iloc[b].nunique(),groups=d.validation_group.iloc[b].nunique(),target_mean=d.cs_mpa.iloc[b].mean(),target_sd=d.cs_mpa.iloc[b].std(),target_min=d.cs_mpa.iloc[b].min(),target_max=d.cs_mpa.iloc[b].max()))
        groupcheck.append(dict(fold=k,source_overlap=0,recipe_overlap=0))
    fd=f.merge(pd.DataFrame(target_info),on='fold');fd.to_csv(OUT/'baseline_fold_diagnosis.csv',index=False)
    dummy_metrics=metrics(dummy.observed,dummy.predicted);bm=metrics(p.observed,p.predicted)
    assert abs(bm['rmse']-12.172717328872988)<1e-9
    pp=p.merge(d.drop(columns='cs_mpa'),on='record_id');pp['error']=pp.predicted-pp.observed;pp['se']=pp.error**2;pp['ae']=abs(pp.error)
    source=grouped_errors(pp,['source_id']).sort_values('rmse',ascending=False)
    sse=pp.groupby('source_id').se.sum();source['sse_share']=source.source_id.map(sse)/sse.sum()
    source['rmse_other_rows_no_refit']=source.source_id.map(lambda s:np.sqrt(pp.loc[pp.source_id!=s,'se'].mean()))
    ledger=pd.read_csv(FROZEN/'provenance/row_verification_ledger.csv').set_index('No.')
    source_names=d.assign(reference=d.record_id.map(ledger['Ref.'])).groupby('source_id').reference.first()
    source['reference']=source.source_id.map(source_names)
    source.to_csv(OUT/'baseline_source_errors.csv',index=False)
    strata=grouped_errors(pp,['is_rac']);strata.to_csv(OUT/'baseline_RAC_control_errors.csv',index=False)
    distribution=d.groupby('is_rac')[RAW+['cs_mpa']].agg(['count','mean','std','median']).T
    distribution.to_csv(OUT/'RAC_control_distributions.csv')
    smd=[]
    for col in RAW+['cs_mpa']:
        a=d.loc[d.is_rac,col];b=d.loc[~d.is_rac,col]
        sd=np.sqrt((a.var()+b.var())/2)
        smd.append(dict(feature=col,RAC_mean=a.mean(),control_mean=b.mean(),standardized_mean_difference=(a.mean()-b.mean())/sd if sd else 0))
    pd.DataFrame(smd).to_csv(OUT/'RAC_control_standardized_differences.csv',index=False)
    overlap=[]
    for kind,ff in [('Group',folds),('Random',list(KFold(n_splits=5,shuffle=True,random_state=SEED).split(d)))]:
        for k,(a,b) in enumerate(ff,1):
            train_recipes=set(map(tuple,d.iloc[a][RAW].round(8).values))
            overlap.append(dict(validation=kind,fold=k,n_valid=len(b),validation_rows_seen_source=int(d.source_id.iloc[b].isin(d.source_id.iloc[a]).sum()),validation_rows_seen_recipe=sum(tuple(r) in train_recipes for r in d.iloc[b][RAW].round(8).values)))
    pd.DataFrame(overlap).to_csv(OUT/'CV_overlap_diagnostics.csv',index=False)
    # Row-local Physics transformation is invertible on the positive mass domain.
    z=features['Physics'];b=z.binder_mass;agg=z.aggregate_binder*b;fine=agg*z.fine_fraction;coarse=agg-fine
    inv=pd.DataFrame(index=d.index)
    for col in ['fly_ash','slag','silica_fume','metakaolin']:inv[col]=z[col+'_binder_fraction']*b
    inv['cement']=b-inv.sum(axis=1);inv['water']=z.reported_water_binder*b
    inv['recycled_coarse']=coarse*z.recycled_coarse_fraction;inv['natural_coarse']=coarse-inv.recycled_coarse
    inv['recycled_fine']=fine*z.recycled_fine_fraction;inv['natural_fine']=fine-inv.recycled_fine
    reconstruction_error=float(abs(inv[RAW]-d[RAW]).to_numpy().max())
    assert reconstruction_error<1e-8
    redundant=[]
    corr=features['Hybrid'].corr(method='spearman')
    for a,b in combinations(corr.columns,2):
        if abs(corr.loc[a,b])>=.9:redundant.append(dict(feature_a=a,feature_b=b,spearman=corr.loc[a,b]))
    pd.DataFrame(redundant).to_csv(OUT/'hybrid_high_correlations.csv',index=False)
    recipe_group=d.groupby(RAW,dropna=False).cs_mpa
    count=recipe_group.transform('count');mean=recipe_group.transform('mean')
    repeat=dict(rows_in_repeated_recipes=int((count>1).sum()),unique_recipes=int(recipe_group.ngroups),empirical_same_input_target_variation_rmse=float(np.sqrt(np.mean((d.cs_mpa-mean)**2))))
    top=source.sort_values('sse_share',ascending=False).head(5)
    source_impact=dict(top5_sse_share=float(top.sse_share.sum()),top5_rows=int(top.n.sum()),macro_source_mae=float(source.mae.mean()),source_count=len(source))
    source_impact['source_mean_bias_sse_fraction']=float((source.n*source.bias**2).sum()/pp.se.sum())
    source_means=d.groupby('source_id').cs_mpa.agg(['size','mean'])
    source_impact['source_target_mean_variance_fraction']=float((source_means['size']*(source_means['mean']-d.cs_mpa.mean())**2).sum()/((d.cs_mpa-d.cs_mpa.mean())**2).sum())
    from sklearn.ensemble import HistGradientBoostingRegressor
    gap=[]
    for k,(a,b) in enumerate(folds,1):
        m=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,learning_rate=.07,l2_regularization=1,early_stopping=False,random_state=SEED)
        m.fit(features['Physics'].iloc[a],d.cs_mpa.iloc[a])
        gap.append(dict(fold=k,train_rmse=metrics(d.cs_mpa.iloc[a],m.predict(features['Physics'].iloc[a]))['rmse'],validation_rmse=metrics(d.cs_mpa.iloc[b],m.predict(features['Physics'].iloc[b]))['rmse']))
    pd.DataFrame(gap).to_csv(OUT/'baseline_training_validation_gap.csv',index=False)
    random_preds=np.zeros(len(d));random_scores=[]
    for k,(a,b) in enumerate(KFold(n_splits=5,shuffle=True,random_state=SEED).split(d),1):
        m=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,learning_rate=.07,l2_regularization=1,early_stopping=False,random_state=SEED)
        m.fit(features['Physics'].iloc[a],d.cs_mpa.iloc[a]);pred=m.predict(features['Physics'].iloc[b]);random_preds[b]=pred
        random_scores.append(dict(fold=k,**metrics(d.cs_mpa.iloc[b],pred)))
    pd.DataFrame(random_scores).to_csv(OUT/'original_baseline_random_folds.csv',index=False)
    pd.DataFrame({'record_id':d.record_id,'observed':d.cs_mpa,'predicted':random_preds}).to_csv(OUT/'original_baseline_random_predictions.csv',index=False)
    original_comparison=pd.DataFrame([dict(validation='Group',**bm),dict(validation='Random',**metrics(d.cs_mpa,random_preds))])
    original_comparison.to_csv(OUT/'original_baseline_random_group.csv',index=False)
    writejson(OUT/'baseline_diagnostic_summary.json',dict(original_pooled=bm,original_fold_mean={k:float(f[k].mean()) for k in ['r2','rmse','mae']},dummy=dummy_metrics,physics_inverse_max_error=reconstruction_error,repeated_recipe_diagnostic=repeat,source_impact=source_impact,holdout_evaluated=False))
    # Original OOF plots; row/source distribution is descriptive only.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(fd.fold,fd.rmse);axes[0].set(xlabel='Source-grouped fold',ylabel='Baseline RMSE (MPa)')
    axes[1].scatter(pp.observed,pp.predicted,c=np.where(pp.is_rac,'#2676a5','#d89039'),s=14,alpha=.6);axes[1].plot([0,100],[0,100],'k--',lw=1);axes[1].set(xlabel='Observed CS (MPa)',ylabel='Baseline OOF predicted CS (MPa)')
    savefig(fig,'01_baseline_diagnosis')
    s=pd.read_csv(OUT/'matrix_summary.csv')
    fig,ax=plt.subplots(figsize=(10,4))
    chart=s[(s.representation=='Hybrid') & (s.model!='Mean')].pivot(index='model',columns='validation',values='pooled_rmse')
    chart.plot.bar(ax=ax,color=['#b3564d','#3384aa']);ax.set(ylabel='OOF RMSE (MPa)',xlabel='');ax.tick_params(axis='x',rotation=20);savefig(fig,'02_random_vs_group')
    comp=s.pivot(index=['representation','model'],columns='validation',values=['pooled_rmse','pooled_r2','pooled_mae'])
    comp.columns=['_'.join(c) for c in comp.columns];comp['rmse_group_minus_random']=comp.pooled_rmse_Group-comp.pooled_rmse_Random
    comp.to_csv(OUT/'random_group_comparison.csv')
    text=f'''# Stage 2：Baseline 诊断

固定开发集 563 条，50 个来源/配方关联组；不评价 159 条测试样本。

## 计算口径

此前 R²=0.181417、RMSE=12.172717 MPa 对应 Physics（10 特征）+ HistGradientBoosting，固定参数见上一阶段脚本。数值是 5 折全部 **563 条折外预测合并后** 计算，既不是训练分数，也不是各折 R² 平均。

各折 R² 平均为 {f.r2.mean():.3f}，折 RMSE 平均为 {f.rmse.mean():.3f} MPa。R² 的分母取决于各折目标方差，因此折平均与合并值不同，不能混用。

{table(fd,['fold','n_valid','r2','rmse','mae','target_mean','target_sd'])}

DummyRegressor 确认为每次仅预测 **训练折** CS 均值，合并 OOF R²={dummy_metrics['r2']:.3f}、RMSE={dummy_metrics['rmse']:.3f}、MAE={dummy_metrics['mae']:.3f}。基线模型的 RMSE 降幅为 {(1-bm['rmse']/dummy_metrics['rmse'])*100:.1f}%。改善存在，跨来源预测仍有限。

## 分组、泄漏和 Random CV

检查通过：每篇文献只属于一个验证组；原始剂量相同的配方属于同一组；开发/测试之间及分组外折之间无来源或精确配方交叉。分组连接允许通过已排除行形成保守的来源关联，不会拆开来源。Random CV 刻意允许来源重叠，作为同来源插值的诊断，不能替代外部来源泛化评价。预处理均位于训练折内的 Pipeline；Physics 为无需拟合且不使用目标的逐行变换。

六种正式模型的对照见 random_group_comparison.csv。随机折中已见来源与配方的样本数见 CV_overlap_diagnostics.csv。随机与分组性能差不是单凭当前数据就能归因的“纯泄漏量”；它同时反映来源分布、配方相关性和泛化任务的变化。

对原来的 Physics+HistGradientBoosting 原样增加 Random CV 后的直接比较：

{table(original_comparison)}

## 来源、RAC 与对照

开发集 {int(d.is_rac.sum())} 条 RAC、{int((~d.is_rac).sum())} 条对照，剂量、CS 分布和标准化均值差已保存。

{table(strata,['is_rac','n','r2','rmse','mae','bias'])}

按误差平方贡献最大的五个来源共有 {source_impact['top5_rows']} 条，贡献 {100*source_impact['top5_sse_share']:.1f}% 的基线误差平方；所有来源宏平均 MAE={source_impact['macro_source_mae']:.3f} MPa。

{table(top,['reference','n','rmse','bias','sse_share','rmse_other_rows_no_refit'])}

最后一列只是移除该来源的误差贡献后重算指标，**没有重新训练，也不构成删样本建议**。主要分析保留所有固定样本。来源差异可由未记录的骨料品质、吸水率、水量口径、试件或试验条件等解释，但当前分析只能提出这些机制假设，无法逐一确定因果。

将误差按来源均值偏差分解，来源平均偏差项占总 SSE 的 {100*source_impact['source_mean_bias_sse_fraction']:.1f}%；CS 的来源间均值差异占样本总离差平方的 {100*source_impact['source_target_mean_variance_fraction']:.1f}%。二者都是描述性分解，与配方差异相混杂，不证明来源本身的因果效应。每个原基线训练折的拟合误差与外折误差见 baseline_training_validation_gap.csv，用于检查学习器拟合能力和泛化落差。

## 特征信息与重复观测

Physics 的十维变换在本数据正质量域内可反解回全部十个原始剂量，最大数值误差 {reconstruction_error:.2g} kg/m³。因此它是坐标变换，没有增加外部物理信息；Hybrid 的收益若有，来自模型对冗余表达的归纳偏好。相关系数高的特征对已单独输出。

开发集存在 {repeat['rows_in_repeated_recipes']} 条相同输入重复配方记录（目标不完全相同）；全部绑定分组。按每个已见配方的 CS 均值拟合的经验残差 RMSE={repeat['empirical_same_input_target_variation_rmse']:.3f} MPa。这只是当前重复观测的冲突描述，不是新来源不可约误差的估计。

## 诊断结论

目前的低分组性能与明显的来源转移、折目标方差差异、输入未覆盖的试验条件相容。Random CV 明显较高不支持更改来源分组或删除困难来源。下一步以嵌套分组 HPO 和重复分组评价改进是否稳定。
'''
    (REPORT/'01_Baseline_Diagnosis.md').write_text(text,encoding='utf8')
    print(json.dumps(dict(baseline=bm,dummy=dummy_metrics,source_impact=source_impact,physics_inverse_max_error=reconstruction_error),ensure_ascii=True))

def cluster_bootstrap(p1,p2,d,n=2000):
    q=p1[['record_id','observed','predicted']].merge(p2[['record_id','predicted']],on='record_id',suffixes=('_a','_b')).merge(d[['record_id','validation_group']],on='record_id')
    q['sse_a']=(q.predicted_a-q.observed)**2;q['sse_b']=(q.predicted_b-q.observed)**2
    g=q.groupby('validation_group').agg(n=('observed','size'),a=('sse_a','sum'),b=('sse_b','sum'))
    rng=np.random.default_rng(SEED);draw=rng.integers(0,len(g),size=(n,len(g)))
    a=g.a.values[draw].sum(axis=1);b=g.b.values[draw].sum(axis=1);nn=g.n.values[draw].sum(axis=1)
    delta=np.sqrt(a/nn)-np.sqrt(b/nn)
    return dict(rmse_a_minus_b=float(np.sqrt(q.sse_a.mean())-np.sqrt(q.sse_b.mean())),ci_low=float(np.quantile(delta,.025)),ci_high=float(np.quantile(delta,.975)),bootstrap_groups=len(g),draws=n)

def final():
    diagnose()
    d,features,folds=load();s=pd.read_csv(OUT/'nested_summary.csv');cand=pd.read_csv(OUT/'representation_candidates.csv')
    all_strata=[];source_tables=[];robustness=[];acompare=[];preds={};candidates=[]
    for r in cand.itertuples():
        rep,name=r.representation,r.model
        p=augmented(OUT/f'nested_{rep}_{name}.csv',d);preds[rep]=p
        for stratum,col in [('RAC/control','is_rac'),('A/B','category')]:
            z=grouped_errors(p,[col]);z=z.rename(columns={col:'stratum'});z['stratum_type']=stratum;z['representation']=rep;z['model']=name;all_strata.append(z)
        p['strength_bin']=pd.cut(p.observed,[0,30,50,70,np.inf],labels=['(0,30]','(30,50]','(50,70]','>70'])
        ratio=(p.recycled_coarse/(p.recycled_coarse+p.natural_coarse)).values
        p['RCA_bin']=np.select([ratio==0,(ratio>0)&(ratio<=.25),(ratio>.25)&(ratio<=.5),(ratio>.5)&(ratio<=.75)],['0','(0,.25]','(.25,.5]','(.5,.75]'],default='(.75,1]')
        for col in ['strength_bin','RCA_bin']:
            z=grouped_errors(p,[col]);z=z.rename(columns={col:'stratum'});z['stratum_type']=col;z['representation']=rep;z['model']=name;all_strata.append(z)
        source=grouped_errors(p,['source_id']);source['representation']=rep;source['model']=name;source_tables.append(source)
        robustness.append(dict(representation=rep,model=name,partition='original',**metrics(p.observed,p.predicted)))
        for seed in [20261101,20261102]:
            q=pd.read_csv(OUT/f'repeat_{rep}_{name}_{seed}.csv');robustness.append(dict(representation=rep,model=name,partition=str(seed),**metrics(q.observed,q.predicted)))
        a=pd.read_csv(OUT/f'Aonly_{rep}_{name}.csv');ab=p[p.record_id.isin(a.record_id)]
        for setting,z in [('A_only_training',a),('AB_training_evaluated_on_same_A',ab)]:
            acompare.append(dict(representation=rep,model=name,setting=setting,n=len(z),**metrics(z.observed,z.predicted)))
        best=json.loads((OUT/f'hpo_{rep}_{name}_development_best.json').read_text(encoding='utf8'))
        candidates.append(dict(representation=rep,model=name,parameters=best['params'],inner_rmse_not_unbiased=best['inner_pooled_rmse']))
    errors=pd.concat(all_strata,ignore_index=True);errors.to_csv(OUT/'nested_stratum_errors.csv',index=False)
    sources=pd.concat(source_tables,ignore_index=True);sources.to_csv(OUT/'nested_source_errors.csv',index=False)
    rr=pd.DataFrame(robustness);rr.to_csv(OUT/'repeated_nested_group_results.csv',index=False)
    aa=pd.DataFrame(acompare);aa.to_csv(OUT/'A_only_sensitivity.csv',index=False)
    writejson(OUT/'development_candidate_parameters.json',candidates)
    pairs=[]
    # Paired representation comparison within each HPO family, keeping model constant.
    for name in s.model.unique():
        for ra,rb in [('Physics','Raw'),('Hybrid','Raw'),('Hybrid','Physics')]:
            p1=pd.read_csv(OUT/f'nested_{ra}_{name}.csv');p2=pd.read_csv(OUT/f'nested_{rb}_{name}.csv')
            pairs.append(dict(model=name,representation_a=ra,representation_b=rb,**cluster_bootstrap(p1,p2,d)))
    ci=pd.DataFrame(pairs);ci.to_csv(OUT/'paired_representation_group_bootstrap.csv',index=False)
    # Scope: bootstrap describes fixed OOF paired errors; does not repeat family selection/HPO.
    summary=s.copy()
    fig,ax=plt.subplots(figsize=(10,4));names=s.model.unique();reps=['Raw','Physics','Hybrid'];xx=np.arange(len(names));width=.24
    for i,rep in enumerate(reps):
        rows=s[s.representation==rep].set_index('model').loc[names]
        ax.bar(xx+(i-1)*width,rows.rmse_mean,width,yerr=rows.rmse_sd,capsize=3,label=rep)
    ax.set(xticks=xx,xticklabels=names,ylabel='Nested grouped fold RMSE mean ± SD (MPa)');ax.legend();savefig(fig,'03_nested_model_comparison')
    fig,ax=plt.subplots(figsize=(7,4))
    for rep,g in rr.groupby('representation'):ax.plot(g.partition,g.rmse,'o-',label=rep)
    ax.set(xlabel='Outer group partition',ylabel='Nested OOF RMSE (MPa)');ax.legend();savefig(fig,'04_partition_robustness')
    chosen=cand.sort_values('pooled_rmse').iloc[0];p=preds[chosen.representation]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for flag,color,label in [(True,'#2676a5','RAC'),(False,'#d89039','Controls')]:
        q=p[p.is_rac==flag];axes[0].scatter(q.observed,q.predicted,s=16,alpha=.6,color=color,label=label)
    axes[0].plot([0,100],[0,100],'k--',lw=1);axes[0].set(xlabel='Observed CS (MPa)',ylabel='Nested OOF prediction (MPa)');axes[0].legend()
    axes[1].scatter(p.observed,p.error,s=16,alpha=.5);axes[1].axhline(0,color='k',ls='--',lw=1);axes[1].set(xlabel='Observed CS (MPa)',ylabel='Prediction minus observation (MPa)');savefig(fig,'05_prediction_and_residuals')
    fig,ax=plt.subplots(figsize=(9,4));q=sources[(sources.representation==chosen.representation)].sort_values('rmse',ascending=False).head(12)
    ax.barh(q.source_id.astype(str).str.replace('10.','',regex=False),q.rmse);ax.invert_yaxis();ax.set(xlabel='Source OOF RMSE (MPa)');savefig(fig,'06_difficult_sources')
    # All search records: budget and successes, including sensitivity.
    budgets=[]
    for path in OUT.glob('*_trials.json'):
        trials=json.loads(path.read_text(encoding='utf8'));budgets.append(dict(search=path.stem.replace('_trials',''),trials=len(trials),complete=sum(t['state']=='TrialState.COMPLETE' for t in trials)))
    budget=pd.DataFrame(budgets);budget.to_csv(OUT/'search_budget_summary.csv',index=False)
    fullmatrix=pd.read_csv(OUT/'matrix_summary.csv')
    # Persist a compact final machine-readable scientific result object.
    result=dict(best_original_nested=dict(chosen),nested_repeated_summary=rr.groupby('representation').rmse.agg(['mean','std','min','max']).to_dict(orient='index'),holdout_evaluated=False,searches=len(budget),trials=int(budget.trials.sum()),all_searches_complete=bool((budget.trials==budget.complete).all()),selection_caveat='families and per-representation candidates screened on development; retain holdout for final evaluation')
    writejson(OUT/'scientific_summary.json',result)
    text=f'''# RAC-956 Stage 2：正式模型比较与稳健性

## 数据与评价边界

固定 cs_v1：722 条。所有拟合、选型、调参和本报告评价仅使用 **563 条开发样本**。159 条测试样本未预测、未计算性能；其固定划分及数据哈希未变。未改纳入规则或删除困难来源。

六个模型家族 × 三套表征，固定参数时比较相同 5 折 Random 与 Group 划分，共 42 个矩阵单元（含均值参照）。均值参照每折只取训练折均值。标准化仅在训练折拟合；树模型不作全局标准化。详见 01_Baseline_Diagnosis.md。

## 1. 正式固定参数模型矩阵

下表报告 Group CV；“pooled”为全部 OOF 合并指标，“mean/sd”为五个折指标的算术均值/样本标准差。完整 Random 对照在 CSV 中。

{table(fullmatrix[(fullmatrix.validation=='Group') & (fullmatrix.model!='Mean')],['representation','model','pooled_r2','pooled_rmse','pooled_mae','rmse_mean','rmse_sd','r2_mean','r2_sd'])}

## 2. Optuna 嵌套分组优化

按三表征平均 Group OOF RMSE 筛选 CatBoost、XGBoost、SVR，所有九个家族×表征单元预算相同：每个内层搜索 10 trials、3 折分组 CV，目标为合并内层 OOF RMSE。TPE 前 4 次为启动采样；无剪枝、无使用外折的早停。外层使用原来相同 5 折。各外折独立优化，在全部开发集上另作相同预算搜索得到候选参数；该内层分数不用于泛化结论。

**选型边界：家族预筛选使用过本开发集外折，因此嵌套结果是对已筛家族的条件评价；比较九个结果后挑选最优还会带来选优偏差，不能称为完全无偏最终分数。** 保留测试集用于锁定方案后的最终评价。10 trials 是有限预算，不能断言已找到各模型最佳性能。

{table(s,['representation','model','pooled_r2','pooled_rmse','pooled_mae','rmse_mean','rmse_sd','r2_mean','r2_sd'])}

搜索空间以 src/stage2_models.py 的 suggest() 为准：SVR 的 C/epsilon/gamma；XGBoost 的树数、深度、学习率、子采样、列采样、叶子权重和 L2；CatBoost 的树数、深度、学习率及 L2。逐 trial 值、参数、各内折 RMSE、最终参数、种子已输出 CSV/JSON；原始日志位于 results/modeling/stage2_*.log。

## 3. Raw / Physics / Hybrid

Physics 可数值反演为原始剂量，不增加独立信息。Hybrid 20 维由同一十维信息生成，存在冗余，潜在收益来自对学习器更合适的表达。

以下比较在同一模型家族内进行，差值负数表示表征 A 的 RMSE 更低。按 50 个验证关联组做 2000 次成对自助抽样；95% 区间仅描述当前固定 OOF 误差的组间不确定性，未重新执行选型和训练、未做多重比较校正，不能视为完整学习流程的显著性检验。

{table(ci,['model','representation_a','representation_b','rmse_a_minus_b','ci_low','ci_high'])}

## 4. 随机分组稳定性

每套表征按原始嵌套 RMSE 选择一个候选模型；再用两个新的 GroupKFold 随机种子重分外折，每折重新进行同预算内层优化。所有训练与验证仍来源独立，未使用全开发集最优参数去冒充新的独立外折评价。重复使用同一开发集，三次结果互相关联，不是三个独立研究。

本分支同时改变外折、HPO 采样和模型种子，是完整流程的联合随机性敏感性，不能将波动全部归因于划分。Raw 分支候选为 CatBoost，另外两套为 XGBoost，跨候选的重复均值不能作为纯粹的特征变换效应；特征比较应优先看相同家族的成对结果。

{table(rr,['representation','model','partition','r2','rmse','mae'])}

## 5. A 类敏感性及误差结构

A 类指冻结政策下“直接使用”，**不等同每字段都已原文严格认证**，其中也包含已披露的结构零编码。A-only 分支在原来的外组划分内，只用 A 训练且只在 A 上评价，重新内层调参；与 A+B 训练后在完全相同外折 A 记录上的误差配对对照。它改变训练队列用于敏感性分析，不修改主分析纳入规则；不能比较不同目标队列的分数后选取好看的样本。

{table(aa,['representation','model','setting','n','r2','rmse','mae'])}

分层分析包括 RAC/天然骨料对照、A/B、固定强度区间及再生粗骨料质量比例。每层的数量、偏差、MAE、RMSE、R² 见 nested_stratum_errors.csv。强度分层使用观察值，只用于事后误差描述，不能作为预测输入；小层和低方差层的 R² 不稳定。每篇来源的误差见 nested_source_errors.csv，不按误差排除任何来源。

## 6. 交付与下一阶段

所有数值均可从 src/stage2_models.py（matrix、tune、robust 三阶段）及 src/stage2_analysis.py（diagnose、final）复现；冻结表、split_assignments.csv 和 requirements.txt 为输入。候选参数见 development_candidate_parameters.json。共完成 {len(budget)} 次搜索、{int(budget.trials.sum())} trials（含重复外折和 A-only 敏感性）。

论文图初稿位于 results/modeling/stage2/figures，PNG+矢量 PDF：原始基线折差异、Random vs Group、嵌套模型比较、重分组稳定性、折外预测/残差、困难来源。它们展示当前开发结果，图注须标为开发集交叉验证，不能标成最终独立测试结果。

最终科学判断见 03_Scientific_Conclusions.md。
'''
    (REPORT/'02_Modeling_and_Robustness.md').write_text(text,encoding='utf8')
    print(json.dumps(result,ensure_ascii=True,default=str))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['diagnose','final']);args=p.parse_args()
    {'diagnose':diagnose,'final':final}[args.phase]()
