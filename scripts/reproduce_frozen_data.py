"""Reconstruct six CSVs in a disposable directory; original files unchanged."""
from pathlib import Path
import shutil,tempfile,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='rac956-freeze-') as name:
    d=Path(name)
    for f in ['src','data/raw','data/frozen/cs_v1/provenance']:(d/f).mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'src/freeze_cs_data_v1.py',d/'src/freeze_cs_data_v1.py')
    shutil.copy2(ROOT/'data/raw/RACdatadriven.xlsx',d/'data/raw/RACdatadriven.xlsx')
    for p in (ROOT/'data/frozen/cs_v1/provenance').glob('*.csv'):shutil.copy2(p,d/'data/frozen/cs_v1/provenance'/p.name)
    subprocess.run([sys.executable,str(d/'src/freeze_cs_data_v1.py')],check=True)
    for p in (ROOT/'data/frozen/cs_v1').glob('*.csv'):assert p.read_bytes()==(d/'data/frozen/cs_v1'/p.name).read_bytes(),p.name
print('PASS isolated raw-to-freeze byte comparison')
