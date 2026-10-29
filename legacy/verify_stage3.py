"""Final Stage 3 gate: existing prediction parity, SHAP arithmetic and holdout protection."""
from stage3_compute import *
import importlib.metadata, platform
from PIL import Image
import xml.etree.ElementTree as ET

def main():
    d,features,folds=load();dev=set(d.record_id);all_audits=[]
    old_verification=json.loads((STAGE2/'verification.json').read_text())
    assert digest(FROZEN/'modeling.csv')==old_verification['modeling_sha256']
    assert digest(BASE/'split_assignments.csv')==old_verification['split_sha256']
    for family in ['XGBoost','CatBoost']:
        for rep,x in features.items():
            seen=[];old=pd.read_csv(STAGE2/f'nested_{rep}_{family}.csv').set_index('record_id')
            for k,(train,valid) in enumerate(folds,1):
                stem=f'{family}_{rep}_fold{k}';m=json.loads((CACHE/(stem+'_manifest.json')).read_text())
                assert set(m['training_ids'])==set(d.record_id.iloc[train])
                assert set(m['validation_ids'])==set(d.record_id.iloc[valid])
                assert digest(STAGE2/f'hpo_{rep}_{family}_outer{k}_best.json')==m['parameter_sha256']
                assert digest(MODELS/(stem+'.joblib'))==m['model_sha256']
                a=np.load(CACHE/(stem+'.npz'));ids=a['record_id'];seen.extend(ids.tolist())
                assert set(ids)<=dev and a['features'].tolist()==x.columns.tolist()
                assert np.allclose(a['values'],x.set_axis(d.record_id).loc[ids].to_numpy())
                assert np.max(abs(a['prediction']-old.loc[ids].predicted))<1e-6
                assert np.max(abs(a['phi'].sum(axis=1)+a['base']-a['prediction']))<1e-3
                all_audits.append(m)
            assert len(seen)==563 and set(seen)==dev and len(set(seen))==563
    assert len(all_audits)==30
    p=pd.read_csv(OUT/'material_block_shapley_values.csv')
    assert len(p)==563*6*3 and set(p.record_id)==dev
    for (family,rep),sub in p.groupby(['family','representation']):
        assert sub.groupby('record_id').size().eq(3).all()
        value=sub.groupby('record_id').shap.sum()+sub.groupby('record_id').reference.first()
        old=pd.read_csv(STAGE2/f'nested_{rep}_{family}.csv').set_index('record_id')
        assert abs(value-old.predicted.loc[value.index]).max()<1e-4
    bg=pd.read_csv(OUT/'material_block_training_backgrounds.csv')
    for k,(a,b) in enumerate(folds,1):
        ids=bg.loc[bg.fold==k,'record_id'];assert len(ids)==64 and ids.is_unique
        assert set(ids)<=set(d.record_id.iloc[a]) and not set(ids)&set(d.record_id.iloc[b])
    permutation=pd.read_csv(OUT/'permutation_all_repeats.csv');assert len(permutation)==4640
    assert np.isfinite(permutation.delta_rmse).all()
    native=pd.read_csv(OUT/'cross_fitted_shap_long.csv');assert len(native)==45040 and set(native.record_id)==dev
    lock=json.loads((OUT/'final_analysis_lock.json').read_text());assert lock['holdout_evaluated'] is False
    assert set(lock['development_ids'])==dev
    assert lock['modeling_sha256']==digest(FROZEN/'modeling.csv')
    assert lock['split_sha256']==digest(BASE/'split_assignments.csv')
    for m in lock['comparators']:
        assert digest(ROOT/m['model_file'])==m['model_sha256']
        fitted=joblib.load(ROOT/m['model_file'])
        assert all(fitted.get_params()[k]==v for k,v in m['parameters'].items())
    # Existing Stage 2 optimization budget must stay exactly unchanged.
    budget=pd.read_csv(STAGE2/'search_budget_summary.csv');assert len(budget)==99 and budget.trials.sum()==990
    figure_checks=[]
    for image in sorted((OUT/'figures').glob('Figure_*.png')):
        stem=image.with_suffix('');svg=stem.with_suffix('.svg');pdf=stem.with_suffix('.pdf')
        assert svg.exists() and pdf.exists()
        content=svg.read_text(encoding='utf8');assert '<text' in content
        tree=ET.fromstring(content);width=float(tree.attrib['width'].removesuffix('pt'))*25.4/72
        assert abs(width-183)<.1
        with Image.open(image) as im:
            dpi=im.info.get('dpi',(0,0));assert min(dpi)>599
            figure_checks.append(dict(name=image.stem,width_mm=width,pixels=list(im.size),dpi=list(dpi),svg_editable_text=True))
    assert len(figure_checks)==8
    # Inspect PDF text when a PDF reader is already available; no new dependencies.
    try:
        from pypdf import PdfReader
        for pdf in (OUT/'figures').glob('Figure_*.pdf'):
            assert len(''.join(p.extract_text() or '' for p in PdfReader(pdf).pages))>30
        pdf_text='PASS'
    except ImportError:pdf_text='not_checked_reader_unavailable; SVG text and configured PDF TrueType verified separately'
    requirements={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions(path=[str(ROOT/'.deps')])}
    (OUT/'requirements_full.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(requirements.items(),key=lambda t:t[0].lower()))+'\n')
    result=dict(status='PASS',python=platform.python_version(),development_n=563,holdout_n=159,holdout_evaluated=False,outer_models_checked=30,native_shap_contributions_checked=45040,group_game_values_checked=len(p),permutation_runs_checked=4640,stage2_trials_unchanged=990,max_native_shap_additivity_error=max(m['shap_additivity_max_error'] for m in all_audits),max_stage2_oof_prediction_difference=max(m['oof_prediction_max_difference'] for m in all_audits),figures=figure_checks,pdf_text_check=pdf_text,modeling_sha256=digest(FROZEN/'modeling.csv'),split_sha256=digest(BASE/'split_assignments.csv'),scripts_sha256={p.name:digest(p) for p in (ROOT/'src').glob('*stage3*.py')})
    writejson(OUT/'verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
