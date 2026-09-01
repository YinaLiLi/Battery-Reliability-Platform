"""Compare canonical measurement ordering anomalies with the official pickles."""
import argparse, json, pickle
from pathlib import Path
import pyarrow.parquet as pq

def main(root, raw, output):
    anomalies=[]
    for path in sorted((Path(root)/'cycle_measurements').glob('*.parquet')):
        prior=None
        for batch in pq.ParquetFile(path).iter_batches(columns=['battery_id','cycle_index','sample_index','time_in_s']):
            for row in batch.to_pylist():
                key=(row['battery_id'],row['cycle_index'],row['sample_index'])
                kind=None
                if prior and key <= prior[:3]: kind='duplicate_or_nonmonotonic_sample_index'
                elif prior and key[:2] == prior[:2] and row['time_in_s'] is not None and prior[3] is not None and row['time_in_s'] < prior[3]: kind='source_time_reset'
                if kind:
                    anomalies.append({'battery_id':key[0],'cycle_index':key[1],'previous_sample_index':prior[2],'sample_index':key[2],'previous_time_in_s':prior[3],'time_in_s':row['time_in_s'],'anomaly_type':kind})
                prior=(*key,row['time_in_s'])
    for item in anomalies:
        cell=pickle.load(open(Path(raw)/(item['battery_id']+'.pkl'),'rb'))
        cycle=next(c for c in cell['cycle_data'] if int(c.get('cycle_number')) == item['cycle_index'])
        times=list(cycle.get('time_in_s') or [])
        i=item['sample_index']; item['source_pickle_position']=i
        item['source_neighboring_times']=times[max(0,i-1):i+2]
        item['source_matches_canonical']=item['time_in_s'] == (times[i] if i < len(times) else None)
        item['classification']='source_data_issue' if item['source_matches_canonical'] else 'normalizer_issue'
        item['recommended_treatment']='preserve_source_time_and_order_by_sample_index' if item['classification']=='source_data_issue' else 'repair_normalizer'
    Path(output).write_text(json.dumps(anomalies,indent=2))
    print(json.dumps(anomalies,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/processed/matr'); p.add_argument('--raw',default='data/raw/batterylife/MATR'); p.add_argument('--output',default='data/processed/matr/anomalies.json'); a=p.parse_args(); main(a.root,a.raw,a.output)
