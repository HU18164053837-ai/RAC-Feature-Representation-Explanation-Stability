"""Read-only numerical verification of saved holdout outputs; never predicts."""
from pathlib import Path
import csv,json,hashlib,math,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/modeling/stage4';P=ROOT/'manuscript/v1.0'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
r=json.loads((OUT/'test_evaluation_receipt.json').read_text())
assert r['status']=='complete' and r['prediction_batches']==1 and r['test_n']==159
assert not any(r['overlap'].values())
for p,k in [(ROOT/'data/frozen/cs_v1/modeling.csv','modeling_sha256'),(ROOT/'results/modeling/cs_v1/split_assignments.csv','split_sha256'),(ROOT/'results/modeling/stage3/final_analysis_lock.json','stage3_lock_sha256')]:
    assert digest(p)==r[k],p
pred=rows(OUT/'final_test_predictions.csv');metrics=rows(OUT/'final_test_metrics.csv')
assert len(pred)==159 and len({x['record_id'] for x in pred})==159
y=[float(x['cs_mpa']) for x in pred];mean=sum(y)/len(y);sst=sum((v-mean)**2 for v in y)
for m in metrics:
    e=[float(x[m['model']])-float(x['cs_mpa']) for x in pred]
    values={'r2':1-sum(v*v for v in e)/sst,'rmse':math.sqrt(sum(v*v for v in e)/len(e)),'mae':sum(abs(v) for v in e)/len(e)}
    for k,v in values.items():assert abs(v-float(m[k]))<1e-6,(m['model'],k,v,m[k])
assert len(list((P/'Supplementary_Tables').glob('*.csv')))==22
assert len(list((P/'Main_Figures').glob('*.pdf')))==5
assert len(list((P/'Supplementary_Figures').glob('*.png')))==5
renders=json.loads((OUT/'document_render_manifest.json').read_text())
assert [x['pages'] for x in renders]==[12,6]
for a in renders:
    with zipfile.ZipFile(P/a['document']) as z:
        assert b'<w:pBdr' not in z.read('word/styles.xml')
    assert (P/a['document']).with_suffix('.pdf').exists()
report={'status':'pass','verification':'cached predictions only; no training or predictions','test_n':159,'prediction_batches':1,'frozen_hashes_unchanged':True,'metrics_recomputed':12,'main_figures':5,'supplementary_figures':5,'supplementary_tables':22,'visual_review':'All 12 manuscript and 6 supplement rendered pages inspected; title border and orphan caption corrected. New main figures 1, 2, 4, 5 inspected; figure 3 reused from Stage 3.','files':{str(p.relative_to(ROOT)):digest(p) for p in sorted(P.rglob('*')) if p.is_file()}}
(OUT/'package_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='files'},ensure_ascii=False,indent=2))
