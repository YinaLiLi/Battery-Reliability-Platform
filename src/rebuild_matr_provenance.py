"""Rebuild the small provenance manifest without rewriting canonical data."""
import pickle
from pathlib import Path
from matr_data import PROVENANCE_SCHEMA, _fingerprint, _write_rows, build_provenance

raw=Path('data/raw/batterylife/MATR'); output=Path('data/processed/matr/matr_provenance.parquet')
cells=[]
for path in sorted(raw.glob('*.pkl')):
    data=path.read_bytes(); cell=pickle.loads(data); cell['_source_fingerprint']=_fingerprint(data); cells.append((path.name,cell))
_write_rows(output, build_provenance(cells), PROVENANCE_SCHEMA)
print(output)
