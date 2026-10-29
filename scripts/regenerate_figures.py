"""Read saved tables; write separate display outputs."""
from pathlib import Path
import sys, inspect, shutil, json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import stage3_figures as g,stage4_figures as h
import pandas as pd, numpy as np
g.LABEL['reported_water_binder']='Dose-derived W/B'
out=ROOT/'generated_figures';out.mkdir(exist_ok=True);g.FIG=out;h.MAIN=out
def devload():
    d=pd.read_csv(g.OUT/'development_residual_source_data.csv');features=g.representations(d)
    ids=d.record_id.to_numpy();folds=[]
    for k in range(1,6):
        m=json.loads((g.OUT/'explanations'/f'XGBoost_Raw_fold{k}_manifest.json').read_text())
        folds.append((np.flatnonzero(np.isin(ids,m['training_ids'])),np.flatnonzero(np.isin(ids,m['validation_ids']))))
    return d,features,folds
def sources():
    d,f,_=devload();s=pd.read_csv(g.OUT/'source_distribution_shift_errors.csv').sort_values(['cs_mean','source_id']).reset_index(drop=True)
    s['source_code']=[f'S{i+1:02d}' for i in range(len(s))]
    return d,f,s
g.load=devload;g.source_tables=sources
g.figure_c();g.figure_d();h.design();h.performance();h.common_material();h.stability()
g.figure_c(family='CatBoost',name='CatBoost_sensitivity')
body=inspect.getsource(g.figure_b).replace('def figure_b():','def revised_sources():').replace("'Water/binder'","'Dose-derived W/B'")
exec(body,g.__dict__);g.revised_sources()
g.supplementary()
shutil.copy2(ROOT/'figures/supplementary/Figure_S5_repeated_group_partitions.png',out/'Figure_S5_repeated_group_partitions.png')
print('Saved-table displays in generated_figures/')
