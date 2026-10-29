"""Publication draft figures A–E; real development observations only, Python."""
from stage3_compute import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import string
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.titlesize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
COLORS={'Raw':'#6A8CAF','Physics':'#6AA89F','Hybrid':'#BA8DA8'}
LABEL={'cement':'Cement','fly_ash':'Fly ash','slag':'Slag','silica_fume':'Silica fume','metakaolin':'Metakaolin','natural_coarse':'Natural coarse','recycled_coarse':'Recycled coarse','natural_fine':'Natural fine','recycled_fine':'Recycled fine','water':'Reported water','binder_mass':'Binder mass','reported_water_binder':'Reported water/binder','aggregate_binder':'Aggregate/binder','fine_fraction':'Fine aggregate fraction','recycled_coarse_fraction':'RCA fraction','recycled_fine_fraction':'RFA fraction','fly_ash_binder_fraction':'Fly ash/binder','slag_binder_fraction':'Slag/binder','silica_fume_binder_fraction':'Silica fume/binder','metakaolin_binder_fraction':'Metakaolin/binder'}

def export(fig,name):
    # Fixed double-column width, editable vector text, high-resolution real-data preview.
    fig.savefig(FIG/(name+'.pdf'),facecolor='white')
    fig.savefig(FIG/(name+'.svg'),facecolor='white')
    fig.savefig(FIG/(name+'.png'),dpi=600,facecolor='white')
    plt.close(fig)

def panel(ax,letter,title):
    ax.set_title(letter+'  '+title,loc='left',fontweight='bold',pad=7)

def figure_a():
    fig,axs=plt.subplots(1,2,figsize=(7.204724,3.307087),layout='constrained',gridspec_kw={'width_ratios':[.85,1.4]})
    q=pd.read_csv(STAGE2/'original_baseline_random_group.csv').set_index('validation').loc[['Random','Group']]
    axs[0].bar([0,1],q.rmse,color=['#9BABB6','#51748C'],width=.55)
    for i,r in enumerate(q.itertuples()):axs[0].text(i,r.rmse+.35,f'{r.rmse:.2f}\nR² = {r.r2:.3f}',ha='center',fontsize=7)
    axs[0].set(xticks=[0,1],xticklabels=['Random','Source-grouped'],ylabel='Pooled OOF RMSE (MPa)',ylim=(0,16));panel(axs[0],'a','Identical original baseline')
    s=pd.read_csv(STAGE2/'matrix_summary.csv');s=s[s.model!='Mean']
    t=s.pivot(index='model',columns=['validation','representation'],values='pooled_rmse')
    values=np.column_stack([t['Group',r]-t['Random',r] for r in COLORS])
    im=axs[1].imshow(values,cmap='Blues',vmin=0,vmax=max(7,float(values.max())),aspect='auto')
    axs[1].set(xticks=range(3),xticklabels=list(COLORS),yticks=range(len(t)),yticklabels=t.index)
    for i in range(len(t)):
        for j in range(3):axs[1].text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if values[i,j]>4 else 'black',fontsize=7)
    fig.colorbar(im,ax=axs[1],label='Group − Random RMSE (MPa)',shrink=.8);panel(axs[1],'b','All fixed-parameter comparisons')
    export(fig,'Figure_A_validation_gap')

def source_tables():
    d,features,_=load();s=pd.read_csv(OUT/'source_distribution_shift_errors.csv').sort_values(['cs_mean','source_id']).reset_index(drop=True)
    s['source_code']=[f'S{i+1:02d}' for i in range(len(s))]
    s[['source_code','source_id','n','cs_mean']].to_csv(OUT/'figure_source_codes.csv',index=False)
    return d,features,s

def figure_b():
    d,features,s=source_tables();res=pd.read_csv(OUT/'development_residual_source_data.csv');g=pd.read_csv(OUT/'validation_group_shift_errors.csv')
    fig=plt.figure(figsize=(7.204724,6.811024),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[1.3,1]);axs=[fig.add_subplot(grid[i,j]) for i in range(2) for j in range(2)]
    rng=np.random.default_rng(SEED)
    for i,row in enumerate(s.itertuples()):
        q=res[res.source_id==row.source_id]
        axs[0].scatter(q.cs_mpa,np.full(len(q),i)+rng.uniform(-.2,.2,len(q)),s=4,c=np.where(q.is_rac,'#557F9D','#C19A64'),alpha=.7,linewidths=0,rasterized=True)
    ticks=np.arange(0,len(s),5)
    axs[0].set(yticks=ticks,yticklabels=s.source_code.iloc[ticks],xlabel='Observed CS (MPa)',ylabel='Source (ordered by mean CS)',ylim=(len(s),-1));panel(axs[0],'a',f'All {len(s)} development sources')
    keys=['binder_mass','reported_water_binder','recycled_coarse_fraction','aggregate_binder'];x=features['Physics'][keys].copy();x['source_id']=d.source_id
    means=x.groupby('source_id')[keys].mean().loc[s.source_id]
    z=(means-features['Physics'][keys].mean())/features['Physics'][keys].std()
    vmax=max(2,float(abs(z.to_numpy()).max()));im=axs[1].imshow(z,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    axs[1].set(xticks=range(4),xticklabels=['Binder','Water/binder','RCA fraction','Aggregate/binder'],yticks=ticks,yticklabels=s.source_code.iloc[ticks]);axs[1].tick_params(axis='x',rotation=35)
    fig.colorbar(im,ax=axs[1],label='Source mean, development z-score',shrink=.8);panel(axs[1],'b','Mix-proportion heterogeneity')
    for i,row in enumerate(s.itertuples()):
        q=res[res.source_id==row.source_id].Physics_residual
        axs[2].scatter(q,np.full(len(q),i)+rng.uniform(-.2,.2,len(q)),s=4,color='#557F9D',alpha=.5,linewidths=0,rasterized=True)
        axs[2].plot([q.mean()],[i],marker='|',color='black',ms=4)
    axs[2].axvline(0,color='gray',lw=.6);axs[2].set(yticks=ticks,yticklabels=s.source_code.iloc[ticks],xlabel='Prediction − observed CS (MPa)',ylabel='Same source order',ylim=(len(s),-1));panel(axs[2],'c','Physics + XGBoost residuals')
    sc=axs[3].scatter(g.wasserstein_shift,g.Physics_rmse,c=g.RAC_fraction,cmap='cividis',vmin=0,vmax=1,s=10+2*np.sqrt(g.n),alpha=.85,linewidths=.3,edgecolors='white')
    a=pd.read_csv(OUT/'shift_error_associations.csv').query("representation == 'Physics' and shift_metric == 'wasserstein_shift'").iloc[0]
    axs[3].text(.03,.97,f'n = 50 validation groups\nSpearman ρ = {a.spearman:.2f}\n95% interval [{a.ci_low:.2f}, {a.ci_high:.2f}]',transform=axs[3].transAxes,va='top',fontsize=6)
    axs[3].set(xlabel='Normalized Wasserstein shift',ylabel='Group OOF RMSE (MPa)');fig.colorbar(sc,ax=axs[3],label='RAC fraction',shrink=.8);panel(axs[3],'d','No clear shift–error relationship')
    export(fig,'Figure_B_sources_and_errors')

def swarm_y(shap):
    # Deterministic density stacking; all observations are drawn, none subsampled.
    bins=np.digitize(shap,np.linspace(float(np.min(shap))-1e-6,float(np.max(shap))+1e-6,35))
    y=np.zeros(len(shap))
    for b in np.unique(bins):
        ix=np.flatnonzero(bins==b);offset=np.arange(len(ix));offset=np.where(offset%2==0,offset/2,-(offset+1)/2)
        y[ix]=offset*.017
    mx=max(.32,float(abs(y).max()));return y*.32/mx

def figure_c(family='XGBoost',name='Figure_C_SHAP_comparison'):
    st=pd.read_csv(OUT/'feature_rank_stability.csv');long=pd.read_csv(OUT/'cross_fitted_shap_long.csv')
    fig,axs=plt.subplots(2,3,figsize=(7.204724,5.708661),layout='constrained',gridspec_kw={'height_ratios':[.85,1.35]})
    for col,rep in enumerate(COLORS):
        q=st[(st.family==family)&(st.representation==rep)].sort_values('mean_importance',ascending=False).head(8).iloc[::-1]
        y=np.arange(len(q));axs[0,col].barh(y,q.mean_importance,xerr=q.importance_sd,color=COLORS[rep],height=.65,error_kw={'elinewidth':.5,'capsize':1.5})
        axs[0,col].set(yticks=y,yticklabels=[LABEL[c] for c in q.feature],xlabel='Mean |SHAP| (MPa)');panel(axs[0,col],string.ascii_lowercase[col],rep+f' ({10 if rep!="Hybrid" else 20} features)')
        for i,c in enumerate(q.feature):
            a=long[(long.family==family)&(long.representation==rep)&(long.feature==c)]
            lo,hi=np.quantile(a.value,[.05,.95]);v=np.clip((a.value-lo)/(hi-lo if hi>lo else 1),0,1)
            axs[1,col].scatter(a.shap,i+swarm_y(a.shap.values),c=v,cmap='coolwarm',vmin=0,vmax=1,s=3,alpha=.65,linewidths=0,rasterized=True)
        axs[1,col].axvline(0,color='gray',lw=.5);axs[1,col].set(yticks=y,yticklabels=[LABEL[c] for c in q.feature],xlabel='Cross-fitted SHAP (MPa)');panel(axs[1,col],string.ascii_lowercase[col+3],'OOF rows (n=563)')
    bar=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='coolwarm'),ax=axs[1,:],orientation='horizontal',shrink=.5,pad=.08,aspect=45);bar.set_ticks([0,1],labels=['Low','High']);bar.set_label('Feature value (within-feature normalized color)')
    export(fig,name)

def figure_d():
    long=pd.read_csv(OUT/'cross_fitted_shap_long.csv');d,_,_=load();rca=(d.recycled_coarse/(d.recycled_coarse+d.natural_coarse));rca.index=d.record_id
    fig,axs=plt.subplots(3,2,figsize=(7.204724,6.535433),layout='constrained')
    pairs={'Raw':['cement','water'],'Physics':['binder_mass','reported_water_binder'],'Hybrid':['cement','reported_water_binder']};source=[]
    for row,(rep,cols) in enumerate(pairs.items()):
        for col,c in enumerate(cols):
            ax=axs[row,col];q=long[(long.family=='XGBoost')&(long.representation==rep)&(long.feature==c)].copy()
            ax.scatter(q.value,q.shap,c=q.record_id.map(rca),cmap='cividis',vmin=0,vmax=1,s=5,alpha=.65,linewidths=0,rasterized=True)
            q['bin']=pd.qcut(q.value,8,duplicates='drop')
            b=q.groupby('bin',observed=True).agg(x=('value','median'),y=('shap','median'),n=('shap','size'))
            ax.plot(b.x,b.y,color='black',lw=.8,marker='o',ms=2,label='Pooled bin median')
            b=b.reset_index();b['representation']=rep;b['feature']=c;source.append(b)
            unit=' (kg/m³)' if c in ['cement','water','binder_mass'] else ''
            ax.axhline(0,color='gray',lw=.5);ax.set(xlabel=LABEL[c]+unit,ylabel='SHAP (MPa)');panel(ax,string.ascii_lowercase[row*2+col],rep+' — '+LABEL[c])
    axs[0,0].legend(fontsize=6);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='cividis'),ax=axs,orientation='horizontal',shrink=.5,aspect=45);cb.set_label('Recycled coarse aggregate fraction')
    pd.concat(source).to_csv(OUT/'dependence_figure_bin_medians.csv',index=False)
    export(fig,'Figure_D_dependence')

def figure_e():
    st=pd.read_csv(OUT/'feature_rank_stability.csv');pair=pd.read_csv(OUT/'pairwise_fold_rank_correlations.csv');pfi=pd.read_csv(OUT/'permutation_rank_stability.csv')
    fig=plt.figure(figsize=(7.204724,7.125984),layout='constrained');grid=fig.add_gridspec(2,6,height_ratios=[1.5,1]);top=[fig.add_subplot(grid[0,i*2:i*2+2]) for i in range(3)];bottom=[fig.add_subplot(grid[1,:3]),fig.add_subplot(grid[1,3:])]
    for ax,rep,letter in zip(top,COLORS,'abc'):
        q=st[(st.family=='XGBoost')&(st.representation==rep)].sort_values('mean_rank').iloc[::-1];y=np.arange(len(q))
        ax.errorbar(q.mean_rank,y,xerr=q.rank_sd,fmt='o',color=COLORS[rep],ms=2.5,elinewidth=.6,capsize=1.5)
        ax.set(yticks=y,yticklabels=[LABEL[c] for c in q.feature],xlabel='Mean rank ± fold SD',xlim=(.4,len(q)+1));panel(ax,letter,rep)
    for i,rep in enumerate(COLORS):
        for family,off,marker in [('XGBoost',-.12,'o'),('CatBoost',.12,'s')]:
            q=pair[(pair.representation==rep)&(pair.family==family)].spearman.to_numpy();xs=np.full(len(q),i+off)+np.linspace(-.045,.045,len(q))
            bottom[0].scatter(xs,q,s=12,color=COLORS[rep],marker=marker,alpha=.7,label=family if i==0 else None)
    bottom[0].set(xticks=range(3),xticklabels=list(COLORS),ylabel='Pairwise fold Spearman ρ',ylim=(.5,1.015));bottom[0].legend(fontsize=6);panel(bottom[0],'d','Ranking concordance (10 fold pairs)')
    for i,rep in enumerate(COLORS):
        q=pfi[(pfi.family=='XGBoost')&(pfi.representation==rep)&(pfi.kind=='material_within_source')].set_index('feature').loc[list(BLOCKS)]
        bottom[1].errorbar(np.arange(3)+(i-1)*.13,q.mean_delta_rmse,yerr=q.outer_fold_sd,fmt='o',ms=3,capsize=2,elinewidth=.7,color=COLORS[rep],label=rep)
    bottom[1].axhline(0,color='gray',lw=.5);bottom[1].set(xticks=range(3),xticklabels=list(BLOCKS),ylabel='Material-block ΔRMSE (MPa)');bottom[1].legend(fontsize=6);panel(bottom[1],'e','Within-source, recomputed features')
    export(fig,'Figure_E_explanation_stability')

def supplementary():
    g=pd.read_csv(OUT/'validation_group_shift_errors.csv');r=pd.read_csv(OUT/'shift_error_associations.csv')
    fig,axs=plt.subplots(3,3,figsize=(7.204724,6.102362),layout='constrained')
    for i,rep in enumerate(COLORS):
        for j,key in enumerate(['wasserstein_shift','mean_standardized_displacement','nearest_distance']):
            a=r[(r.representation==rep)&(r.shift_metric==key)].iloc[0];ax=axs[i,j]
            ax.scatter(g[key],g[rep+'_rmse'],s=7,color=COLORS[rep],alpha=.7)
            ax.text(.03,.97,f'ρ={a.spearman:.2f}; q={a.q_BH:.2f}',transform=ax.transAxes,va='top',fontsize=6)
            ax.set(xlabel={'wasserstein_shift':'Wasserstein shift','mean_standardized_displacement':'Mean displacement','nearest_distance':'Nearest-recipe distance'}[key],ylabel=rep+' RMSE (MPa)');panel(ax,string.ascii_lowercase[i*3+j],'50 validation groups')
    export(fig,'Figure_S1_all_shift_associations')
    figure_c('CatBoost','Figure_S2_CatBoost_SHAP')
    block=pd.read_csv(OUT/'material_block_shapley_stability.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.204724,3.307087),layout='constrained')
    for i,rep in enumerate(COLORS):
        q=block[(block.family=='XGBoost')&(block.representation==rep)].set_index('concept').loc[list(BLOCKS)]
        axs[0].errorbar(np.arange(3)+(i-1)*.13,q.mean_abs_shap,yerr=q.outer_fold_sd,fmt='o',ms=3,capsize=2,elinewidth=.7,color=COLORS[rep],label=rep)
    axs[0].set(xticks=range(3),xticklabels=list(BLOCKS),ylabel='Common-input block |Shapley| (MPa)');axs[0].legend(fontsize=6);panel(axs[0],'a','Exact three-player empirical game')
    q=block.pivot(index=['family','representation'],columns='concept',values='top1_frequency').loc[:,list(BLOCKS)]
    im=axs[1].imshow(q,aspect='auto',cmap='Blues',vmin=0,vmax=1)
    axs[1].set(xticks=range(3),xticklabels=q.columns,yticks=range(len(q)),yticklabels=[a+' / '+b for a,b in q.index])
    for i in range(len(q)):
        for j in range(3):axs[1].text(j,i,f'{q.iloc[i,j]:.0%}',ha='center',va='center',fontsize=6,color='white' if q.iloc[i,j]>.6 else 'black')
    panel(axs[1],'b','Top-1 frequency across five outer folds');export(fig,'Figure_S3_common_input_material_Shapley')

if __name__=='__main__':
    figure_a();figure_b();figure_c();figure_d();figure_e();supplementary();print('EXPORTED A–E, S1–S2: editable PDF/SVG and 600-dpi PNG')
