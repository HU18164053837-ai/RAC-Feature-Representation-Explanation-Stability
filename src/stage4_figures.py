"""Assemble existing results into five main and five supplementary figure bundles."""
from stage2_models import ROOT,OUT as S2,BASE,FROZEN
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S3=ROOT/'results/modeling/stage3';S4=ROOT/'results/modeling/stage4'
PAPER=ROOT/'manuscript/v1.0';MAIN=PAPER/'Main_Figures';SUP=PAPER/'Supplementary_Figures'
for p in [MAIN,SUP]:p.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.titlesize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
C={'Raw':'#6A8CAF','Physics':'#6AA89F','Hybrid':'#BA8DA8'}

def export(fig,name):
    fig.savefig(MAIN/(name+'.pdf'));fig.savefig(MAIN/(name+'.svg'));fig.savefig(MAIN/(name+'.png'),dpi=600);plt.close(fig)

def label(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=7)

def design():
    fig,axs=plt.subplots(1,2,figsize=(7.204724,3.937008),layout='constrained')
    ax=axs[0];ax.set_axis_off();label(ax,'a  Frozen cohort')
    boxes=[(.5,.9,'Public workbook\n980 candidate records'),(.5,.65,'A: 603   B: 119   C: 258\nIncluded A + B: 722'),(.25,.34,'Development: 563\n54 sources / 50 groups'),(.77,.34,'Holdout: 159\n13 sources / 13 groups'),(.5,.07,'536 RAC + 186 natural-aggregate controls\nNo source or exact-recipe overlap across split')]
    for x,y,text in boxes:ax.text(x,y,text,ha='center',va='center',fontsize=7,bbox={'boxstyle':'round,pad=.5','fc':'#EFF3F5','ec':'#AAB9C1','lw':.6},transform=ax.transAxes)
    for start,end in [((.5,.83),(.5,.72)),((.46,.57),(.27,.43)),((.55,.57),(.75,.43))]:ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':.7})
    ax=axs[1];ax.set_axis_off();label(ax,'b  Fixed comparison and explanation design')
    texts=['Raw 10  |  Physics 10  |  Hybrid 20\nSame reported material information',
        'Development only\n6 families → 3 screened families\n5 outer / 3 inner grouped folds; 990 trials total',
        'Matched explanations\nXGBoost + CatBoost, 5 outer models each\nNative TreeSHAP; common material games',
        'Locked XGBoost × 3 representations\nRefit on 563 development rows\nOne batch evaluation of 159 holdout rows']
    ys=[.86,.62,.36,.1]
    for y,text in zip(ys,texts):ax.text(.5,y,text,ha='center',va='center',fontsize=7,bbox={'boxstyle':'round,pad=.5','fc':'#F3F2F6','ec':'#B9ADC4','lw':.6},transform=ax.transAxes)
    for a,b in zip(ys[:-1],ys[1:]):ax.annotate('',xy=(.5,b+.09),xytext=(.5,a-.085),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':.7})
    export(fig,'Figure_1_study_design')

def performance():
    fig,axs=plt.subplots(2,2,figsize=(7.204724,5.511811),layout='constrained')
    q=pd.read_csv(S2/'original_baseline_random_group.csv').set_index('validation').loc[['Random','Group']]
    ax=axs[0,0];ax.bar([0,1],q.rmse,color=['#A5B5BE','#54788E'],width=.5)
    for i,r in enumerate(q.itertuples()):ax.text(i,r.rmse+.25,f'{r.rmse:.2f}; R²={r.r2:.3f}',ha='center',fontsize=6)
    ax.set(xticks=[0,1],xticklabels=['Random','Grouped'],ylabel='Pooled OOF RMSE (MPa)',ylim=(0,15));label(ax,'a  Identical development baseline')
    q=pd.read_csv(S2/'nested_summary.csv').query("model == 'XGBoost'").set_index('representation').loc[list(C)]
    ax=axs[0,1];ax.bar(range(3),q.rmse_mean,yerr=q.rmse_sd,color=list(C.values()),capsize=3,error_kw={'elinewidth':.7})
    ax.set(xticks=range(3),xticklabels=list(C),ylabel='Fold RMSE mean ± SD (MPa)',ylim=(0,17));label(ax,'b  Nested grouped XGBoost')
    t=pd.read_csv(S4/'final_test_metrics.csv').set_index('model').loc[['Raw','Physics','Hybrid','Dummy']]
    ax=axs[1,0];ax.errorbar(range(4),t.rmse,yerr=np.vstack([t.rmse-t.rmse_ci_low,t.rmse_ci_high-t.rmse]),fmt='none',color='black',elinewidth=.7,capsize=3)
    ax.scatter(range(4),t.rmse,c=list(C.values())+['#8A8A8A'],s=18)
    ax.set(xticks=range(4),xticklabels=t.index,ylabel='Holdout RMSE, 95% interval (MPa)',ylim=(0,35));label(ax,'c  Source-disjoint holdout (n=159)')
    z=pd.read_csv(S4/'final_test_paired_contrasts.csv');ax=axs[1,1];ax.axvline(0,color='gray',lw=.6)
    ax.errorbar(z.rmse_a_minus_b,np.arange(3),xerr=np.vstack([z.rmse_a_minus_b-z.ci_low,z.ci_high-z.rmse_a_minus_b]),fmt='o',color='#5C7E91',ms=3,capsize=3,elinewidth=.7)
    ax.set(yticks=range(3),yticklabels=z.representation_a+' − '+z.representation_b,xlabel='Holdout RMSE difference (MPa)',ylim=(-.6,2.6));label(ax,'d  Paired representation contrasts')
    export(fig,'Figure_2_predictive_performance')

def common_material():
    block=pd.read_csv(S3/'material_block_shapley_stability.csv');pfi=pd.read_csv(S3/'permutation_rank_stability.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.204724,3.307087),layout='constrained')
    for i,rep in enumerate(C):
        q=block[(block.family=='XGBoost')&(block.representation==rep)].set_index('concept').loc[['Binder','Water','Aggregate']]
        axs[0].errorbar(np.arange(3)+(i-1)*.13,q.mean_abs_shap,yerr=q.outer_fold_sd,fmt='o',ms=3,capsize=2,elinewidth=.7,color=C[rep],label=rep)
        q=pfi[(pfi.family=='XGBoost')&(pfi.representation==rep)&(pfi.kind=='material_within_source')].set_index('feature').loc[['Binder','Water','Aggregate']]
        axs[1].errorbar(np.arange(3)+(i-1)*.13,q.mean_delta_rmse,yerr=q.outer_fold_sd,fmt='o',ms=3,capsize=2,elinewidth=.7,color=C[rep],label=rep)
    for ax in axs:ax.set(xticks=range(3),xticklabels=['Binder','Water','Aggregate']);ax.legend(fontsize=6)
    axs[0].set(ylabel='Material-block |Shapley| (MPa)');axs[1].set(ylabel='Within-source ΔRMSE (MPa)');axs[1].axhline(0,color='gray',lw=.5)
    label(axs[0],'a  Common-input empirical Shapley');label(axs[1],'b  Recomputed material-block permutation');export(fig,'Figure_4_common_material_attribution')

def stability():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from stage3_figures import LABEL
    st=pd.read_csv(S3/'feature_rank_stability.csv');pair=pd.read_csv(S3/'pairwise_fold_rank_correlations.csv')
    fig=plt.figure(figsize=(7.204724,7.125984),layout='constrained');grid=fig.add_gridspec(2,6,height_ratios=[1.5,1]);top=[fig.add_subplot(grid[0,i*2:i*2+2]) for i in range(3)];bottom=[fig.add_subplot(grid[1,:3]),fig.add_subplot(grid[1,3:])]
    for ax,rep,letter in zip(top,C,'abc'):
        q=st[(st.family=='XGBoost')&(st.representation==rep)].sort_values('mean_rank').iloc[::-1];yy=np.arange(len(q))
        ax.errorbar(q.mean_rank,yy,xerr=q.rank_sd,fmt='o',color=C[rep],ms=2.5,capsize=1.5,elinewidth=.6);ax.set(yticks=yy,yticklabels=[LABEL[c] for c in q.feature],xlabel='Mean rank ± fold SD',xlim=(.4,len(q)+1));label(ax,letter+'  '+rep)
    for i,rep in enumerate(C):
        for family,off,marker in [('XGBoost',-.12,'o'),('CatBoost',.12,'s')]:
            q=pair[(pair.representation==rep)&(pair.family==family)]
            for ax,key in zip(bottom,['spearman','top3_fractional_overlap']):ax.scatter(np.full(len(q),i+off)+np.linspace(-.045,.045,len(q)),q[key],s=12,color=C[rep],marker=marker,label=family if i==0 else None,alpha=.7)
    for ax in bottom:ax.set(xticks=range(3),xticklabels=list(C));ax.legend(fontsize=6)
    bottom[0].set(ylabel='Pairwise fold Spearman ρ',ylim=(.5,1.02));bottom[1].set(ylabel='Fractional Top-3 overlap',ylim=(0,1.05))
    label(bottom[0],'d  Full ranking concordance');label(bottom[1],'e  Leading feature-set overlap');export(fig,'Figure_5_explanation_stability')

def copy_bundle(source,dest):
    for ext in ['pdf','svg','png']:shutil.copy2(source.with_suffix('.'+ext),dest.with_suffix('.'+ext))

if __name__=='__main__':
    design();performance();common_material();stability()
    copy_bundle(S3/'figures/Figure_C_SHAP_comparison',MAIN/'Figure_3_native_SHAP')
    for src,name in [('Figure_B_sources_and_errors','Figure_S1_source_heterogeneity'),('Figure_D_dependence','Figure_S2_native_dependence'),('Figure_S1_all_shift_associations','Figure_S3_shift_associations'),('Figure_S2_CatBoost_SHAP','Figure_S4_CatBoost_sensitivity')]:copy_bundle(S3/'figures'/src,SUP/name)
    # Stage 2 repeated-partition figure was originally PNG only; retain its original format.
    shutil.copy2(S2/'figures/04_partition_robustness.png',SUP/'Figure_S5_repeated_group_partitions.png')
    print('Five main figure bundles and five existing supplementary figures assembled.')
