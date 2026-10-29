"""Reproduce frozen CSV in an isolated workspace copy, never edit original data."""
from pathlib import Path
import os,shutil,subprocess,sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tmp/revision_v11_freeze_reproduction'
T.mkdir(parents=True,exist_ok=True)
for relative in ['src','data/raw','data/frozen/cs_v1/provenance']:(T/relative).mkdir(parents=True,exist_ok=True)
shutil.copy2(ROOT/'src/freeze_cs_data_v1.py',T/'src/freeze_cs_data_v1.py')
shutil.copy2(ROOT/'data/raw/RACdatadriven.xlsx',T/'data/raw/RACdatadriven.xlsx')
for p in (ROOT/'data/frozen/cs_v1/provenance').glob('*.csv'):shutil.copy2(p,T/'data/frozen/cs_v1/provenance'/p.name)
env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'.deps')
r=subprocess.run([sys.executable,str(T/'src/freeze_cs_data_v1.py')],env=env,capture_output=True,text=True,encoding='utf8');assert r.returncode==0,r.stderr
checks={}
for name in ['modeling.csv','classification_all_980.csv','processing_log.csv','existing_issue_dispositions.csv','exclusion_summary.csv','data_dictionary.csv']:
    a=(ROOT/'data/frozen/cs_v1'/name).read_bytes();b=(T/'data/frozen/cs_v1'/name).read_bytes();assert a==b,name;checks[name]=hashlib.sha256(b).hexdigest()
(ROOT/'results/modeling/revision_v1.1/freeze_reproduction_check.json').write_text(json.dumps(dict(status='PASS',isolated_copy=True,originals_untouched=True,byte_identical_files=checks),indent=2),encoding='utf8')
print('PASS: six frozen CSV files reproduced byte-for-byte from raw workbook and evidence snapshots.')
