"""Standard library release gate; never imports model or training code."""
from pathlib import Path
import csv,json,hashlib,math
ROOT=Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
manifest=rows('Github_Manifest_SHA256.csv')
for a in manifest:
    p=ROOT/a['path'];assert p.is_file(),a['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256'],a['path']
d=rows('data/frozen/cs_v1/modeling.csv');s=rows('results/modeling/cs_v1/split_assignments.csv')
assert len(d)==722 and len(s)==722
parts={x['record_id']:x['partition'] for x in s}
dev=[x for x in d if parts[x['record_id']]=='development'];test=[x for x in d if parts[x['record_id']]=='holdout']
assert len(dev)==563 and len(test)==159
for k in ['source_id','validation_group']:assert not {x[k] for x in dev}&{x[k] for x in test}
raw=['cement','fly_ash','slag','silica_fume','metakaolin','natural_coarse','recycled_coarse','natural_fine','recycled_fine','water']
assert not {tuple(round(float(x[k]),8) for k in raw) for x in dev}&{tuple(round(float(x[k]),8) for k in raw) for x in test}
c=rows('data/frozen/cs_v1/classification_all_980.csv');assert len(c)==980
assert {k:sum(x['category']==k for x in c) for k in 'ABC'}==dict(A=603,B=119,C=258)
assert sum(x['is_rac'].lower()=='true' for x in d)==536
r=json.loads((ROOT/'results/modeling/stage4/test_evaluation_receipt.json').read_text())
assert r['prediction_batches']==1 and r['test_n']==159
for p,k in [('data/frozen/cs_v1/modeling.csv','modeling_sha256'),('results/modeling/cs_v1/split_assignments.csv','split_sha256')]:
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==r[k]
p=rows('results/modeling/stage4/final_test_predictions.csv');scores=rows('results/modeling/stage4/final_test_metrics.csv');assert len(p)==159
y=[float(x['cs_mpa']) for x in p];avg=sum(y)/159;sst=sum((v-avg)**2 for v in y)
for a in scores:
    e=[float(x[a['model']])-float(x['cs_mpa']) for x in p]
    v={'r2':1-sum(z*z for z in e)/sst,'rmse':math.sqrt(sum(z*z for z in e)/159),'mae':sum(abs(z) for z in e)/159}
    for k,n in v.items():assert abs(n-float(a[k]))<1e-6,(a['model'],k)
lock=json.loads((ROOT/'results/modeling/stage3/final_analysis_lock.json').read_text())
for a,b in zip(lock['comparators'],r['models']):
    assert a['parameters']==b['parameters'] and a['features']==b['features']
print(json.dumps(dict(status='PASS',manifest_files=len(manifest),matrix=722,development=563,consumed_holdout=159,saved_metrics_checked=12,model_fits=0,new_predictions=0,stdlib_only=True),indent=2))
