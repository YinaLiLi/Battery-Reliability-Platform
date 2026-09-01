"""Fit lineage-safe RUL regressors; SOH is a measured capacity health metric."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from .matr_stage2 import split_audit
    from .rul_predictions import constrain_prediction_row
except ImportError:
    from matr_stage2 import split_audit
    from rul_predictions import constrain_prediction_row

ROOT=Path('data/processed/matr')
TARGET='rul_cycles'
RUL_FEATURES=['cycle_index','internal_resistance_in_ohm','temperature_min_in_C','temperature_max_in_C','charge_time_in_s','prior_discharge_capacity_in_Ah','capacity_slope_10','rolling_capacity_mean_10','temperature_span_in_C','charge_time_delta','voltage_min_in_V','voltage_max_in_V','voltage_mean_in_V','current_mean_in_A','current_abs_max_in_A','charge_capacity_in_Ah','discharge_capacity_in_Ah','capacity_fade_from_prior','coulombic_efficiency','early_cycle_capacity_delta']

def metrics(y,p):
    return {'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**.5),'r2':float(r2_score(y,p))}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--model-version'); args=parser.parse_args()
    evaluated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    model_version=args.model_version or f"matr-rul-xgboost-{evaluated_at.replace(':','').replace('+00:00','z')}"
    provenance=pq.read_table(ROOT/'matr_provenance.parquet').to_pylist(); splits,audit=split_audit(provenance)
    table=ds.dataset(ROOT/'degradation_features',format='parquet').to_table().to_pydict()
    valid=np.asarray([v is not None for v in table[TARGET]]); battery=np.asarray(table['battery_id'])
    indexes={name:valid & np.isin(battery,list(ids)) for name,ids in splits.items()}
    matrix=np.asarray([[np.nan if v is None else v for v in table[col]] for col in RUL_FEATURES],float).T; matrix[~np.isfinite(matrix)]=np.nan
    labels=np.asarray([np.nan if v is None else v for v in table[TARGET]],float)
    models={'ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=1,solver='lsqr')),'random_forest':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=100,min_samples_leaf=10,n_jobs=-1,random_state=42))}
    from xgboost import XGBRegressor
    models['xgboost']=make_pipeline(SimpleImputer(strategy='median'),XGBRegressor(n_estimators=200,max_depth=6,learning_rate=.05,n_jobs=4,random_state=42))
    scores={}
    fitted={}
    for name,model in models.items():
        model.fit(matrix[indexes['train']],labels[indexes['train']]); score={part:metrics(labels[indexes[part]],np.maximum(model.predict(matrix[indexes[part]]),0)) for part in ('train','validation','test')}
        score['generalization_gap']={k:score['validation'][k]-score['train'][k] for k in score['train']}
        cycles=np.asarray(table['cycle_index']); eol=np.asarray(table['eol_cycle']); stages=np.select([cycles<eol*.33,cycles<eol*.67],['early','mid'],default='late'); test_stage=stages[indexes['test']]; pred=np.maximum(model.predict(matrix[indexes['test']]),0)
        score['lifecycle_stage_mae']={stage:float(mean_absolute_error(labels[indexes['test']][test_stage==stage],pred[test_stage==stage])) for stage in ('early','mid','late')}; scores[name]=score; fitted[name]=model
    leakage={'target':TARGET,'predictors':RUL_FEATURES,'excluded_predictors':['eol_cycle','rul_cycles','lineage_group_id','battery_id'],'train_only_preprocessing':True,'battery_overlap':audit['battery_overlap'],'lineage_overlap':audit['lineage_overlap'],'soh_policy':'calculated directly as discharge_capacity_in_Ah / nominal_capacity_in_Ah / SOC_width; not ML-predicted'}
    (ROOT/'rul_model_metrics.json').write_text(json.dumps(scores,indent=2)); (ROOT/'lifecycle_stage_error_report.json').write_text(json.dumps({k:v['lifecycle_stage_mae'] for k,v in scores.items()},indent=2)); (ROOT/'leakage_generalization_audit.json').write_text(json.dumps({'split':audit,**leakage,'model_selection':'xgboost: best validation MAE/RMSE/R2'},indent=2))
    raw_predicted=fitted['xgboost'].predict(matrix)
    rows=[constrain_prediction_row({'model_version':model_version,'dataset':table['dataset'][i],'battery_id':table['battery_id'][i],'cycle_index':table['cycle_index'][i],'predicted_rul_cycles':float(raw_predicted[i]),'prediction_created_at':evaluated_at,'split':next(name for name,ids in splits.items() if table['battery_id'][i] in ids)}) for i in range(len(raw_predicted))]
    pq.write_table(__import__('pyarrow').Table.from_pylist(rows),ROOT/'candidate_predictions.parquet')
    training_metadata={'training_data_version':'MATR / degradation_features','lineage_manifest':'matr_provenance.parquet','training_row_count':int(indexes['train'].sum()),'validation_row_count':int(indexes['validation'].sum()),'test_row_count':int(indexes['test'].sum())}
    evaluation={'model_version':model_version,'model_name':'xgboost_rul_regressor','dataset':'MATR','status':'candidate','evaluated_at':evaluated_at,'metrics_json':json.dumps({'train':scores['xgboost']['train'],'validation':scores['xgboost']['validation'],'test':scores['xgboost']['test'],'lifecycle_stage_mae':scores['xgboost']['lifecycle_stage_mae'],'generalization_gap':scores['xgboost']['generalization_gap']}),'training_metadata_json':json.dumps(training_metadata)}
    pq.write_table(__import__('pyarrow').Table.from_pylist([evaluation]),ROOT/'candidate_model_evaluation.parquet')
if __name__=='__main__': main()
