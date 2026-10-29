"""Relabel existing quantitative figures; no model or outcome recalculation.
Contract: preserve all observations/statistics, nominal dose-derived W/B labels,
183 mm editable PDF/SVG plus 600 dpi PNG, all 20 Hybrid coordinates retained.
"""
from stage2_models import ROOT
import stage3_figures as g
import stage4_figures as h
import pandas as pd,numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.titlesize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
def export_revision(fig,name,destination):
    assert abs(fig.get_size_inches()[0]-7.204724)<1e-4
    fig.savefig(destination/(name+'.pdf'),facecolor='white')
    fig.savefig(destination/(name+'.svg'),facecolor='white')
    fig.savefig(destination/(name+'.png'),dpi=600,facecolor='white')
    plt.close(fig)
g.export=lambda fig,name:export_revision(fig,name,g.FIG)
h.export=lambda fig,name:export_revision(fig,name,h.MAIN)
P=ROOT/'manuscript/v1.1'
g.LABEL['reported_water_binder']='Dose-derived W/B'
def devload():
    d=pd.read_csv(g.OUT/'development_residual_source_data.csv');features=g.representations(d)
    import json
    ids=d.record_id.to_numpy();folds=[]
    for k in range(1,6):
        m=json.loads((g.OUT/'explanations'/f'XGBoost_Raw_fold{k}_manifest.json').read_text())
        folds.append((np.flatnonzero(np.isin(ids,m['training_ids'])),np.flatnonzero(np.isin(ids,m['validation_ids']))))
    return d,features,folds
g.load=devload
def sources():
    d,f,_=devload();s=pd.read_csv(g.OUT/'source_distribution_shift_errors.csv').sort_values(['cs_mean','source_id']).reset_index(drop=True);s['source_code']=[f'S{i+1:02d}' for i in range(len(s))];return d,f,s
g.source_tables=sources
g.FIG=P/'Figures_Main';g.figure_c(name='Figure_3_native_SHAP')
h.MAIN=P/'Figures_Main';h.stability()
g.FIG=P/'Figures_Supplementary'
# Label-only variant of the source profile figure, avoiding writes to historical data.
import inspect
body=inspect.getsource(g.figure_b).replace('def figure_b():','def revised_b():').replace("'Water/binder'","'Dose-derived W/B'").replace("'Figure_B_sources_and_errors'","'Figure_S1_source_heterogeneity'")
exec(body,g.__dict__);g.revised_b()
g.figure_d()
for ext in ['pdf','svg','png']:
    source=g.FIG/('Figure_D_dependence.'+ext);source.replace(g.FIG/('Figure_S2_native_dependence.'+ext))
g.figure_c(family='CatBoost',name='Figure_S4_CatBoost_sensitivity')
print('Updated nominal W/B labels in Figures 3, 5, S1, S2, S4; all others preserved.')
