"""Bounded CS readiness policy v1. No literature retrieval; reproduce from raw + evidence snapshots."""
from pathlib import Path
import hashlib,json,shutil,re,unicodedata
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/frozen/cs_v1'; PROV=OUT/'provenance'
RAW=ROOT/'data/raw/RACdatadriven.xlsx'
FIELDS=['OPC','FA','GGBS','SF','CC','NCA','RCA','NFA','RFA','Water']
NAMES=['cement','fly_ash','slag','silica_fume','metakaolin','natural_coarse','recycled_coarse','natural_fine','recycled_fine','water']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(name):
    p=PROV/name
    if not p.exists():shutil.copy2(ROOT/'results/audit'/name,p)
    return pd.read_csv(p,engine='python')
def main():
    PROV.mkdir(parents=True,exist_ok=True)
    ledger=snapshot('row_verification_ledger.csv').set_index('No.')
    residual=snapshot('corrected_draft_remaining_conflicts.csv')
    fixes=snapshot('applied_numeric_corrections.csv')
    raw=pd.read_excel(RAW,sheet_name='RACdata',header=None)
    x=raw.iloc[3:,:19].copy();x.columns=list(raw.iloc[0,:3])+list(raw.iloc[1,3:19]);x['No.']=x['No.'].astype(int);x=x.set_index('No.')
    assert len(x)==980 and x.index.is_unique
    for f in FIELDS+['CS']:x[f]=pd.to_numeric(x[f],errors='raise')
    log=[];processed={no:[] for no in x.index};reasons={no:[] for no in x.index};flags={no:[] for no in x.index}
    for r in fixes.itertuples():
        if r.field not in FIELDS+['CS']:continue
        old=x.loc[r.no,r.field]
        assert (pd.isna(old) and pd.isna(r.workbook_value)) or abs(float(old)-float(r.workbook_value))<1e-7
        x.loc[r.no,r.field]=r.corrected_value
        processed[r.no].append('evidenced_numeric_correction:'+r.field)
        log.append(dict(no=r.no,field=r.field,before=old,after=r.corrected_value,action='existing_evidenced_correction',basis=r.decision_file))
    # Ordinary public-database dose encoding: blank optional material = no recorded addition.
    # This is an explicit database assumption, not a claim of independent source confirmation.
    for no in x.index:
        for f in ['FA','GGBS','SF','CC','RFA']:
            if pd.isna(x.loc[no,f]):
                x.loc[no,f]=0;log.append(dict(no=no,field=f,before=np.nan,after=0,action='structural_zero_decoding',basis='v1 database dose encoding; not source verification'))
        for f,other in [('NCA','RCA'),('RCA','NCA'),('NFA','RFA')]:
            if pd.isna(x.loc[no,f]) and pd.notna(x.loc[no,other]) and x.loc[no,other]>0:
                x.loc[no,f]=0;log.append(dict(no=no,field=f,before=np.nan,after=0,action='complementary_aggregate_zero',basis=other+' is present'))
        if ledger.loc[no,'verification_level']=='excluded':reasons[no].append('retracted_or_unreliable_source')
        if pd.isna(ledger.loc[no,'candidate_age_days']) or ledger.loc[no,'candidate_age_days']!=28:reasons[no].append('28day_age_not_established')
        if ledger.loc[no,'verification_level']=='partial':
            # All current partial rows concern critical target/mix alignment or unbound experimental response.
            reasons[no].append('critical_CS_age_or_experiment_alignment_unresolved')
        if x.loc[no,FIELDS+['CS']].isna().any():reasons[no].append('missing_required_core_quantity')
    critical=set(FIELDS+['CS','experiment_mapping','natural_aggregate_partition','natural_aggregate_total','recycled_aggregate_partition','row_alignment','source_row_support'])
    dispositions=[]
    for r in residual.itertuples():
        no=int(r.no)
        if r.field in critical:
            reasons[no].append('unresolved_core:'+r.field+':'+r.status);decision='exclude_critical_unresolved'
        elif r.field=='duplicate_mix':
            # Both water representations are ambiguous; no fabricated choice to preserve size.
            reasons[no].append('duplicate_experiment_with_unresolved_water_basis');decision='exclude_ambiguous_duplicate_pair'
        elif r.field=='w/b':
            processed[no].append('drop_nominal_ratio_and_compute_dose_ratios');decision='resolved_by_feature_definition_nominal_ratio_unused'
        elif r.field=='source_identity':
            processed[no].append('repair_source_identity_metadata');decision='source_identity_repaired_using_existing_evidence'
        else:decision='nonblocking_unused_field_or_context'
        dispositions.append(dict(no=no,field=r.field,status=r.status,CS_model_disposition=decision))
    # Deterministic, broad physical validity checks; no outcome-based outlier deletion.
    binder=x[['OPC','FA','GGBS','SF','CC']].sum(axis=1)
    coarse=x[['NCA','RCA']].sum(axis=1);fine=x[['NFA','RFA']].sum(axis=1)
    for no in x.index:
        v=x.loc[no,FIELDS+['CS']]
        if (v.dropna()<0).any() or x.loc[no,'CS']<=0 or x.loc[no,'Water']<=0 or binder.loc[no]<=0 or coarse.loc[no]<=0 or fine.loc[no]<=0:
            reasons[no].append('nonphysical_core_value')
    # Exact duplicates of all selected inputs AND CS collapse to the first original ID.
    # Identical inputs with different CS remain repeated observations and never cross folds.
    eligible=[no for no in x.index if not reasons[no]]
    xy=x.loc[eligible,FIELDS+['CS']].round(8)
    seen={}
    for no,row in xy.iterrows():
        key=tuple(row)
        if key in seen:reasons[no].append('exact_input_target_duplicate_of:'+str(seen[key]))
        else:seen[key]=int(no)
    # Normalize publication identifiers; source identity is grouping metadata, never a predictor.
    source={}
    for no in x.index:
        doi=str(x.loc[no,'doi']).lower().strip().replace('https://','').replace('http://','').replace('doi.org/','')
        if 576<=no<=590:doi='10.1016/j.cemconcomp.2011.05.009'
        if doi in ['nan','none','']:doi=unicodedata.normalize('NFKC',str(x.loc[no,'Ref.'])).lower().strip()
        source[no]=doi
    # Union publication groups sharing exact recipes, preventing cross-source duplicate leakage.
    parent={s:s for s in set(source.values())}
    def find(s):
        while parent[s]!=s:parent[s]=parent[parent[s]];s=parent[s]
        return s
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    recipes={}
    for no in x.index:
        if x.loc[no,FIELDS].isna().any():continue
        key=tuple(x.loc[no,FIELDS].round(8))
        if key in recipes:union(source[no],source[recipes[key]])
        else:recipes[key]=no
    records=[]
    for no in x.index:
        why=sorted(set(reasons[no]));actions=sorted(set(processed[no]));cls='C' if why else 'B' if actions else 'A'
        records.append(dict(record_id=no,category=cls,category_label={'A':'DIRECTLY_USABLE','B':'USABLE_AFTER_PROCESSING','C':'EXCLUDED'}[cls],
            included=cls!='C',exclusion_reasons=';'.join(why),processing_actions=';'.join(actions),
            source_id=source[no],validation_group=find(source[no]),age_days=ledger.loc[no,'candidate_age_days'],
            legacy_source_status=ledger.loc[no,'verification_level'],scope='CS readiness v1 supersedes legacy all-field admission hold'))
    classification=pd.DataFrame(records).set_index('record_id');keep=classification.index[classification.included]
    final=x.loc[keep,FIELDS+['CS']].rename(columns=dict(zip(FIELDS,NAMES))|{'CS':'cs_mpa'})
    final.index.name='record_id';final=classification.loc[keep,['category','source_id','validation_group']].join(final)
    final['age_days']=28;final['is_rac']=(final.recycled_coarse+final.recycled_fine)>0
    assert len(final)>100 and final[NAMES+['cs_mpa']].notna().all().all()
    assert not final[NAMES+['cs_mpa']].round(8).duplicated().any()
    final.to_csv(OUT/'modeling.csv',encoding='utf-8-sig')
    classification.to_csv(OUT/'classification_all_980.csv',encoding='utf-8-sig')
    priority=['retracted_or_unreliable_source','critical_CS_age_or_experiment_alignment_unresolved','duplicate_experiment_with_unresolved_water_basis','unresolved_core','28day_age_not_established','missing_required_core_quantity','nonphysical_core_value','exact_input_target_duplicate_of']
    excluded=classification.loc[classification.category=='C',['exclusion_reasons']].copy()
    excluded['primary']=excluded.exclusion_reasons.map(lambda s:next(k for k in priority if k in s))
    excluded.to_csv(OUT/'exclusion_summary.csv',encoding='utf-8-sig')
    pd.DataFrame(log).to_csv(OUT/'processing_log.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dispositions).to_csv(OUT/'existing_issue_dispositions.csv',index=False,encoding='utf-8-sig')
    definition=[dict(field=n,unit='kg/m3',role='predictor',definition=d) for n,d in zip(NAMES,[
        'Reported cement product dose; may include blended cement, not necessarily pure OPC','Separately reported fly ash','Separately reported slag','Silica fume','CC mapped to metakaolin for included sources','Natural coarse aggregate','Recycled coarse aggregate','Natural fine aggregate','Recycled fine aggregate','Reported mixing-water dose; effective/total basis may vary by study'])]
    definition += [dict(field='cs_mpa',unit='MPa',role='target',definition='28day reported compressive strength; specimen geometry varies by study'),dict(field='source_id',unit='',role='group_only',definition='Publication identity repaired where established; never model input'),dict(field='validation_group',unit='',role='group_only',definition='Connected publications sharing exact recipes'),dict(field='category',unit='',role='metadata',definition='A direct; B after evidenced processing; zero decoding is routine database normalization'),dict(field='age_days',unit='day',role='constant_metadata',definition='28; excluded from predictive inputs'),dict(field='is_rac',unit='',role='stratum',definition='Positive recycled coarse or fine dose; controls retained and reported separately')]
    pd.DataFrame(definition).to_csv(OUT/'data_dictionary.csv',index=False,encoding='utf-8-sig')
    counts=classification.category.value_counts().to_dict()
    manifest=dict(version='cs_v1',date='2026-10-23',initial_candidates=980,A=int(counts.get('A',0)),B=int(counts.get('B',0)),C=int(counts.get('C',0)),modeling_samples=len(final),RAC=int(final.is_rac.sum()),controls=int((~final.is_rac).sum()),sources=final.source_id.nunique(),validation_groups=final.validation_group.nunique(),raw_sha256=sha(RAW),modeling_sha256=sha(OUT/'modeling.csv'),raw_predictors=NAMES,target='cs_mpa',known_critical_issues_remaining_included=0,
        provenance_sha256={p.name:sha(p) for p in PROV.glob('*.csv')},policy='CS-only v1, not certification of all source data. No source retrieval required. Structural zero convention disclosed; SP/nominalwb/other outcomes omitted.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    assert len(classification)==980 and manifest['A']+manifest['B']+manifest['C']==980
    assert final.index.is_unique and (classification.loc[keep,'exclusion_reasons']=='').all()
    print(json.dumps(manifest,ensure_ascii=True))
if __name__=='__main__':main()
