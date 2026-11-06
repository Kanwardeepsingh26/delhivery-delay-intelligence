"""Shared constants and preprocessing used by every part and by predict_delay.py."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
WEIGHT_SENTINEL_LIMIT = 100  # network accepts parcels up to 40 kg; larger = scanner fault


def clean_shipments(raw, medians=None):
    """Clean a raw shipments table. Returns (clean_df, medians)."""
    df = raw.drop_duplicates().copy()                                   # 1. duplicates
    df["service_type"] = df["service_type"].str.strip().str.title()      # 2. service spelling
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")           # 3. payment placeholder
    df["declared_value_inr"] = pd.to_numeric(                           # 4. rupee text -> number
        df["declared_value"].astype("string").str.replace("\u20b9", "", regex=False)
        .str.replace(",", "", regex=False), errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])           # 5. datetime
    df["parcel_weight_kg"] = df["parcel_weight_kg"].astype(float)
    df.loc[df["parcel_weight_kg"] > WEIGHT_SENTINEL_LIMIT, "parcel_weight_kg"] = np.nan  # 6. sentinel
    if medians is None:                                                 # 7. fill gaps with medians
        medians = {c: float(df[c].median()) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].astype(float).fillna(medians[c])
    return df.reset_index(drop=True), medians


def build_features(clean, hubs):
    """Attach hub columns and derive model features. Never touches delivery_hours."""
    df = clean.merge(hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
                     on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
