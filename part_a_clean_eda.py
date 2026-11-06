"""Part A: clean the raw export, build features, write checks / delay evidence / correlations."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES, clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

# "Before" counts are measured after removing duplicates and before any filling
dedup = raw.drop_duplicates()
spell_before = dedup["service_type"].nunique()
w = pd.to_numeric(dedup["parcel_weight_kg"], errors="coerce")
before = {
    "missing_distance_km_before": int(dedup["distance_km"].isna().sum()),
    "missing_parcel_weight_before": int(w.isna().sum()),            # excludes sentinel rows
    "sentinel_weight_rows": int((w > 100).sum()),
    "missing_declared_value_before": int(dedup["declared_value"].isna().sum()),
    "missing_payment_mode_before": int(dedup["payment_mode"].isna().sum()),
}

clean, medians = clean_shipments(raw)
feat = build_features(clean, hubs)
rows_after_hub_merge = len(feat)
promised = feat["promised_hours"]
feat["is_delayed"] = (feat["delivery_hours"] > promised).astype(int)
feat.to_csv("data/modelling_table.csv", index=False)

h = feat["delivery_hours"]
q1, q3 = h.quantile(0.25), h.quantile(0.75)
lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
checks = {
    "raw_rows": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    **before,
    "service_type_spellings_before": spell_before,
    "service_type_spellings_after": feat["service_type"].nunique(),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"],
    "rows_after_hub_merge": rows_after_hub_merge,
    "nulls_in_features_and_target": int(feat[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
    "delayed_count": int(feat["is_delayed"].sum()),
    "delay_rate": feat["is_delayed"].mean(),
    "declared_value_mean_inr": feat["declared_value_inr"].mean(),
    "declared_value_median_inr": feat["declared_value_inr"].median(),
    "log_declared_value_mean": feat["log_declared_value"].mean(),
    "log_declared_value_median": feat["log_declared_value"].median(),
    "hours_q1": q1, "hours_q3": q3,
    "hours_lower_bound": lo, "hours_upper_bound": hi,
    "hours_outlier_count": int(((h < lo) | (h > hi)).sum()),
}
pd.DataFrame({"metric": list(checks), "value": [round(float(v), 4) for v in checks.values()]}) \
    .to_csv("outputs/part_a_checks.csv", index=False)

# Delay evidence by group
feat["distance_band"] = pd.cut(feat["distance_km"], bins=[0, 200, 400, 600, 2000],
                               labels=["0-200", "201-400", "401-600", "601+"], right=True)
parts = []
for dim in ["service_type", "hub_tier", "distance_band"]:
    g = feat.groupby(dim, observed=True).agg(shipments=("is_delayed", "size"), delayed=("is_delayed", "sum"),
                                              delay_rate=("is_delayed", "mean"), mean_hours=("delivery_hours", "mean"))
    g = g.reset_index().rename(columns={dim: "group"})
    g.insert(0, "dimension", dim)
    parts.append(g)
out = pd.concat(parts, ignore_index=True)
out["group"] = out["group"].astype(str)
out[["delay_rate", "mean_hours"]] = out[["delay_rate", "mean_hours"]].round(4)
out.to_csv("outputs/part_a_delay_by_group.csv", index=False)

# Correlations of numeric columns of the modelling table
corr = feat.drop(columns=["distance_band"]).select_dtypes(include=[np.number]).corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv")
print("Part A done: modelling table", feat.shape)
