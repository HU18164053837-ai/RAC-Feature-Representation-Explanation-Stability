"""Stage 2: frozen-development-only comparison, nested HPO and sensitivity.
Usage: python src/stage2_models.py matrix|tune|robust
Outputs are checkpointed by experiment. Holdout targets never enter any fit/score.
"""
from pathlib import Path
import sys, os, json, hashlib, time, argparse, importlib.metadata
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.deps'))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from run_cs_baseline_v1 import representations, metrics, RAW
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED=20261023
OUT=ROOT/'results/modeling/stage2'
OUT.mkdir(parents=True,exist_ok=True)
BASE=ROOT/'results/modeling/cs_v1'
FROZEN=ROOT/'data/frozen/cs_v1'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def writejson(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf8')

def load():
    manifest=json.loads((FROZEN/'manifest.json').read_text(encoding='utf8'))
    assert digest(FROZEN/'modeling.csv')==manifest['modeling_sha256']
    splits=pd.read_csv(BASE/'split_assignments.csv')
    full=pd.read_csv(FROZEN/'modeling.csv')
    # Select development rows before accessing any outcome or constructing features.
    ids=splits.loc[splits.partition=='development','record_id']
    d=full.loc[full.record_id.isin(ids)].copy().reset_index(drop=True)
    assert len(d)==563 and len(splits.loc[splits.partition=='holdout'])==159
    assert full.groupby('source_id').validation_group.nunique().max()==1
    assert full.groupby(RAW,dropna=False).validation_group.nunique().max()==1
    assert full.merge(splits[['record_id','partition']],on='record_id').groupby('validation_group').partition.nunique().max()==1
    features=representations(d)
    fold_by=splits.set_index('record_id').development_fold
    fold=d.record_id.map(fold_by).values
    folds=[(np.flatnonzero(fold!=k),np.flatnonzero(fold==k)) for k in range(1,6)]
    for a,b in folds:check(d,a,b)
    return d,features,folds

def check(d,a,b):
    assert not set(d.iloc[a].validation_group)&set(d.iloc[b].validation_group)
    assert not set(d.iloc[a].source_id)&set(d.iloc[b].source_id)
    assert not set(map(tuple,d.iloc[a][RAW].round(8).values))&set(map(tuple,d.iloc[b][RAW].round(8).values))

def model(name,params=None,seed=SEED):
    p=params or {}
    if name=='Mean':return DummyRegressor(strategy='mean')
    if name=='Ridge':return make_pipeline(StandardScaler(),Ridge(**({'alpha':10.}|p)))
    if name=='SVR':return make_pipeline(StandardScaler(),SVR(**({'C':30.,'epsilon':1.,'gamma':'scale'}|p)))
    if name=='RandomForest':return RandomForestRegressor(**({'n_estimators':150,'min_samples_leaf':2,'n_jobs':1,'random_state':seed}|p))
    if name=='XGBoost':return XGBRegressor(**({'n_estimators':200,'max_depth':3,'learning_rate':.05,'subsample':.85,'colsample_bytree':.9,'reg_lambda':1.,'objective':'reg:squarederror','tree_method':'hist','n_jobs':1,'random_state':seed}|p))
    if name=='LightGBM':return LGBMRegressor(**({'n_estimators':200,'num_leaves':15,'max_depth':-1,'learning_rate':.05,'min_child_samples':15,'verbosity':-1,'n_jobs':1,'random_state':seed,'deterministic':True,'force_col_wise':True}|p))
    if name=='CatBoost':return CatBoostRegressor(**({'iterations':200,'depth':5,'learning_rate':.05,'loss_function':'RMSE','verbose':False,'thread_count':1,'random_seed':seed,'allow_writing_files':False}|p))
    raise ValueError(name)

def cv(d,x,folds,name,params=None,grouped=True,seed=SEED):
    oof=np.full(len(d),np.nan);scores=[]
    for k,(a,b) in enumerate(folds,1):
        if grouped:check(d,a,b)
        t=time.perf_counter();m=model(name,params,seed)
        m.fit(x.iloc[a],d.cs_mpa.iloc[a]);p=m.predict(x.iloc[b]);oof[b]=p
        scores.append(dict(fold=k,n_train=len(a),n_valid=len(b),seconds=time.perf_counter()-t,**metrics(d.cs_mpa.iloc[b],p),
            overlapping_sources=len(set(d.iloc[a].source_id)&set(d.iloc[b].source_id)),
            overlapping_recipes=len(set(map(tuple,d.iloc[a][RAW].round(8).values))&set(map(tuple,d.iloc[b][RAW].round(8).values)))))
    assert np.isfinite(oof).all()
    return pd.DataFrame(scores),oof

def matrix():
    d,features,folds=load()
    random=list(KFold(n_splits=5,shuffle=True,random_state=SEED).split(d))
    names=['Mean','Ridge','SVR','RandomForest','XGBoost','LightGBM','CatBoost']
    configs={n:str(model(n)) for n in names};writejson(OUT/'matrix_model_configs.json',configs)
    for kind,ff in [('Group',folds),('Random',random)]:
        for rep,x in features.items():
            for name in names:
                stem=OUT/f'matrix_{kind}_{rep}_{name}'
                if stem.with_suffix('.csv').exists():continue
                scores,p=cv(d,x,ff,name,grouped=kind=='Group')
                scores.to_csv(str(stem)+'_folds.csv',index=False)
                pd.DataFrame({'record_id':d.record_id,'observed':d.cs_mpa,'predicted':p}).to_csv(stem.with_suffix('.csv'),index=False)
                print('MATRIX',kind,rep,name,metrics(d.cs_mpa,p),flush=True)
    summarize_matrix()

def summarize_matrix():
    result=[]
    for path in OUT.glob('matrix_*_folds.csv'):
        _,kind,rep,name,_=path.stem.split('_')
        f=pd.read_csv(path);p=pd.read_csv(str(path).replace('_folds.csv','.csv'))
        r=dict(validation=kind,representation=rep,model=name,**{'pooled_'+k:v for k,v in metrics(p.observed,p.predicted).items()})
        for key in ['r2','rmse','mae']:r[key+'_mean']=f[key].mean();r[key+'_sd']=f[key].std(ddof=1)
        result.append(r)
    pd.DataFrame(result).to_csv(OUT/'matrix_summary.csv',index=False)

TRIALS=10
def suggest(trial,name):
    if name=='Ridge':return {'alpha':trial.suggest_float('alpha',.01,10000,log=True)}
    if name=='SVR':return {'C':trial.suggest_float('C',1,1000,log=True),'epsilon':trial.suggest_float('epsilon',.05,5,log=True),'gamma':trial.suggest_float('gamma',.001,1,log=True)}
    if name=='RandomForest':return {'n_estimators':150,'max_depth':trial.suggest_categorical('max_depth',[None,4,8,16]),'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,12),'max_features':trial.suggest_float('max_features',.4,1.)}
    if name=='XGBoost':return {'n_estimators':trial.suggest_int('n_estimators',100,400,step=100),'max_depth':trial.suggest_int('max_depth',2,6),'learning_rate':trial.suggest_float('learning_rate',.015,.15,log=True),'min_child_weight':trial.suggest_float('min_child_weight',1,15,log=True),'subsample':trial.suggest_float('subsample',.65,1),'colsample_bytree':trial.suggest_float('colsample_bytree',.6,1),'reg_lambda':trial.suggest_float('reg_lambda',.1,30,log=True)}
    if name=='LightGBM':return {'n_estimators':trial.suggest_int('n_estimators',100,400,step=100),'num_leaves':trial.suggest_int('num_leaves',5,31),'learning_rate':trial.suggest_float('learning_rate',.015,.15,log=True),'min_child_samples':trial.suggest_int('min_child_samples',8,40),'colsample_bytree':trial.suggest_float('colsample_bytree',.6,1),'reg_lambda':trial.suggest_float('reg_lambda',.1,30,log=True)}
    if name=='CatBoost':return {'iterations':trial.suggest_int('iterations',100,400,step=100),'depth':trial.suggest_int('depth',3,7),'learning_rate':trial.suggest_float('learning_rate',.015,.15,log=True),'l2_leaf_reg':trial.suggest_float('l2_leaf_reg',1,30,log=True)}
    raise ValueError(name)

def optimize(d,x,name,tag,seed=SEED):
    path=OUT/(tag+'_best.json')
    if path.exists():return json.loads(path.read_text(encoding='utf8'))
    folds=list(GroupKFold(n_splits=3).split(d,groups=d.validation_group))
    for a,b in folds:check(d,a,b)
    inner_assignment=d[['record_id','source_id','validation_group']].copy()
    inner_assignment['inner_fold']=0
    for k,(a,b) in enumerate(folds,1):inner_assignment.loc[b,'inner_fold']=k
    inner_assignment.to_csv(OUT/(tag+'_inner_assignments.csv'),index=False)
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed,n_startup_trials=4),study_name=tag)
    # Equal trial count across all selected model x representation cells; no early stopping/pruning.
    def objective(t):
        params=suggest(t,name);f,p=cv(d,x,folds,name,params,seed=seed)
        t.set_user_attr('fold_rmse',f.rmse.tolist())
        t.set_user_attr('actual_params',params)
        return metrics(d.cs_mpa,p)['rmse']
    study.optimize(objective,n_trials=TRIALS)
    study.trials_dataframe().to_csv(OUT/(tag+'_trials.csv'),index=False)
    alltrials=[dict(number=t.number,value=t.value,params=t.params,user_attrs=t.user_attrs,state=str(t.state)) for t in study.trials]
    writejson(OUT/(tag+'_trials.json'),alltrials)
    best=dict(params=study.best_trial.user_attrs['actual_params'],inner_pooled_rmse=study.best_value,trials=TRIALS,seed=seed)
    writejson(path,best);return best

def shortlist():
    s=pd.read_csv(OUT/'matrix_summary.csv')
    ranking=s[(s.validation=='Group') & (s.model!='Mean')].groupby('model').pooled_rmse.mean().sort_values()
    selected=ranking.index[:3].tolist()
    writejson(OUT/'hpo_shortlist.json',dict(rule='three families with lowest mean grouped pooled RMSE over representations; development-only screening',ranking=ranking.to_dict(),selected=selected,trials=TRIALS,outer_folds=5,inner_folds=3,selection_caveat='family screening used these development folds; nested estimates condition on screened families'))
    return selected

def tune():
    d,features,folds=load();names=shortlist()
    for rep,x in features.items():
        for name in names:
            target=OUT/f'nested_{rep}_{name}.csv'
            if target.exists():continue
            oof=np.full(len(d),np.nan);scores=[]
            for k,(a,b) in enumerate(folds,1):
                tag=f'hpo_{rep}_{name}_outer{k}'
                best=optimize(d.iloc[a].reset_index(drop=True),x.iloc[a].reset_index(drop=True),name,tag,SEED+k)
                t=time.perf_counter();m=model(name,best['params']);m.fit(x.iloc[a],d.cs_mpa.iloc[a]);p=m.predict(x.iloc[b]);oof[b]=p
                scores.append(dict(fold=k,n_train=len(a),n_valid=len(b),seconds=time.perf_counter()-t,**metrics(d.cs_mpa.iloc[b],p)))
                print('NESTED',rep,name,k,scores[-1],flush=True)
            pd.DataFrame(scores).to_csv(OUT/f'nested_{rep}_{name}_folds.csv',index=False)
            pd.DataFrame({'record_id':d.record_id,'observed':d.cs_mpa,'predicted':oof}).to_csv(target,index=False)
            # All-development search produces candidate settings, never a test estimate.
            optimize(d,x,name,f'hpo_{rep}_{name}_development',SEED)
    summarize_nested()

def summarize_nested():
    rows=[]
    for p in OUT.glob('nested_*_folds.csv'):
        _,rep,name,_=p.stem.split('_');f=pd.read_csv(p);pred=pd.read_csv(str(p).replace('_folds.csv','.csv'))
        r=dict(representation=rep,model=name,**{'pooled_'+k:v for k,v in metrics(pred.observed,pred.predicted).items()})
        for key in ['r2','rmse','mae']:r[key+'_mean']=f[key].mean();r[key+'_sd']=f[key].std(ddof=1)
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'nested_summary.csv',index=False)

def robust():
    d,features,folds=load();s=pd.read_csv(OUT/'nested_summary.csv')
    # Lock one candidate per representation before sensitivity; keep all rows in primary cohort.
    selected=s.loc[s.groupby('representation').pooled_rmse.idxmin()]
    selected.to_csv(OUT/'representation_candidates.csv',index=False)
    for r in selected.itertuples():
        rep,name=r.representation,r.model;x=features[rep]
        # Alternative group partitions use nested inner search, not globally tuned parameters.
        for seed in [20261101,20261102]:
            target=OUT/f'repeat_{rep}_{name}_{seed}.csv'
            if target.exists():continue
            ff=list(GroupKFold(n_splits=5,shuffle=True,random_state=seed).split(d,groups=d.validation_group))
            oof=np.full(len(d),np.nan);scores=[];fold_id=np.zeros(len(d),dtype=int)
            for k,(a,b) in enumerate(ff,1):
                check(d,a,b)
                best=optimize(d.iloc[a].reset_index(drop=True),x.iloc[a].reset_index(drop=True),name,f'repeatHPO_{rep}_{name}_{seed}_fold{k}',seed+k)
                m=model(name,best['params'],seed);m.fit(x.iloc[a],d.cs_mpa.iloc[a]);p=m.predict(x.iloc[b]);oof[b]=p;fold_id[b]=k
                scores.append(dict(fold=k,**metrics(d.cs_mpa.iloc[b],p)))
            pd.DataFrame(scores).to_csv(OUT/f'repeat_{rep}_{name}_{seed}_folds.csv',index=False)
            pd.DataFrame({'record_id':d.record_id,'observed':d.cs_mpa,'predicted':oof,'fold':fold_id}).to_csv(target,index=False)
            print('REPEAT',rep,name,seed,metrics(d.cs_mpa,oof),flush=True)
        # A-only sensitivity: same outer groups, tune only A training rows, evaluate same heldout A.
        target=OUT/f'Aonly_{rep}_{name}.csv'
        if target.exists():continue
        records=[]
        for k,(a,b) in enumerate(folds,1):
            a=a[d.category.iloc[a].values=='A'];b=b[d.category.iloc[b].values=='A'];check(d,a,b)
            best=optimize(d.iloc[a].reset_index(drop=True),x.iloc[a].reset_index(drop=True),name,f'AonlyHPO_{rep}_{name}_fold{k}',SEED+k)
            m=model(name,best['params']);m.fit(x.iloc[a],d.cs_mpa.iloc[a]);p=m.predict(x.iloc[b])
            records.extend(dict(record_id=int(i),observed=float(y),predicted=float(v),fold=k) for i,y,v in zip(d.record_id.iloc[b],d.cs_mpa.iloc[b],p))
        pd.DataFrame(records).to_csv(target,index=False)
        print('AONLY',rep,name,metrics(pd.DataFrame(records).observed,pd.DataFrame(records).predicted),flush=True)
    packages=['numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','catboost','optuna','matplotlib']
    (OUT/'requirements.txt').write_text('\n'.join(p+'=='+importlib.metadata.version(p) for p in packages)+'\n')
    writejson(OUT/'run_manifest.json',dict(seed=SEED,development_n=len(d),holdout_evaluated=False,modeling_sha256=digest(FROZEN/'modeling.csv'),split_sha256=digest(BASE/'split_assignments.csv'),script_sha256=digest(Path(__file__)),trials_per_search=TRIALS,metric_objective='pooled inner group OOF RMSE',no_early_stopping=True))

if __name__=='__main__':
    action=argparse.ArgumentParser();action.add_argument('phase',choices=['matrix','tune','robust']);args=action.parse_args()
    {'matrix':matrix,'tune':tune,'robust':robust}[args.phase]()
