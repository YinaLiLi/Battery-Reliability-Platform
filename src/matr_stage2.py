"""Lineage-safe splits and regression evaluation for MATR Stage 2."""
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLITS=("train","validation","test")

def lineage_split(rows, seed=42):
    """Deterministic 60/20/20 allocation of whole lineage groups."""
    groups={}
    for row in rows: groups.setdefault(row["lineage_group_id"],[]).append(row)
    target={"train":round(len(groups)*.6),"validation":round(len(groups)*.2)}; target["test"]=len(groups)-sum(target.values())
    out={name:set() for name in SPLITS}
    for group, members in sorted(groups.items(), key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()):
        name=min((x for x in SPLITS if len(out[x])<target[x]), key=lambda x:(len(out[x])/target[x], SPLITS.index(x)))
        out[name].update(row["battery_id"] for row in members)
    return out

def split_audit(provenance):
    split=lineage_split(provenance)
    lookup={r['battery_id']:r for r in provenance}
    report={"seed":42,"splits":{},"battery_overlap":False,"lineage_overlap":False}
    lineage_sets={}
    for name, ids in split.items():
        rows=[lookup[x] for x in ids]; lineages={x['lineage_group_id'] for x in rows}; lineage_sets[name]=lineages
        report['splits'][name]={"battery_count":len(ids),"lineage_count":len(lineages),"by_batch":dict(Counter(x['batch_id'] for x in rows)),"by_policy":dict(Counter(x['charge_policy'] for x in rows)),"by_lineage_type":dict(Counter(x['lineage_reason'] for x in rows))}
    report['battery_overlap']=any(split[a]&split[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:])
    report['lineage_overlap']=any(lineage_sets[a]&lineage_sets[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:])
    return split,report

if __name__=='__main__':
    import pyarrow.parquet as pq
    root=Path('data/processed/matr'); provenance=pq.read_table(root/'matr_provenance.parquet').to_pylist(); _,audit=split_audit(provenance)
    (root/'split_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)); print(json.dumps(audit,indent=2))
