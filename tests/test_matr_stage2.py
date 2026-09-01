from src.matr_stage2 import lineage_split
import src.train_matr_models as training


def test_lineage_split_keeps_related_cells_together():
    rows = [
        {"battery_id": "a", "lineage_group_id": "g1", "batch_id": "batch_1", "charge_policy": "p", "eol_cycle": 100},
        {"battery_id": "b", "lineage_group_id": "g1", "batch_id": "batch_1", "charge_policy": "p", "eol_cycle": 100},
        *[{"battery_id": f"x{i}", "lineage_group_id": f"g{i+1}", "batch_id": "batch_2", "charge_policy": "p", "eol_cycle": 200+i} for i in range(8)],
    ]
    split = lineage_split(rows)
    assert next(name for name, ids in split.items() if "a" in ids) == next(name for name, ids in split.items() if "b" in ids)
    assert not (split["train"] & split["validation"] or split["train"] & split["test"] or split["validation"] & split["test"])


def test_final_stage2_trains_rul_only():
    assert not hasattr(training, "SOH_FEATURES")
    assert training.TARGET == "rul_cycles"
