# MATR Stage 2

SOH is the measured cycle-health metric: `max(discharge_capacity_in_Ah) / nominal_capacity_in_Ah / SOC_width`.

It is not an ML target. In MATR, nominal capacity is 1.1 Ah and SOC width is 1.0; current discharge capacity reconstructs SOH exactly, while prior capacity and rolling capacity features have correlations above 0.9996 with SOH. The prior SOH regression result was therefore target reconstruction, not a useful prediction task.

RUL in cycles is the sole ML target. Models use the frozen lineage-disjoint split and causal current/prior-cycle features. `eol_cycle`, `rul_cycles`, battery IDs, and lineage IDs are excluded from predictors. See `data/processed/matr/` for the immutable split, leakage, RUL, and lifecycle reports.
