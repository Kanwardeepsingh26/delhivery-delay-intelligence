"""Fit the final classifier from Part C's configuration and save it with its helper files."""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])

df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

pipe = Pipeline([("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
                 ("model", RandomForestClassifier(random_state=42, n_estimators=int(cfg["n_estimators"]),
                                                  max_depth=max_depth, min_samples_leaf=int(cfg["min_samples_leaf"])))]
                ).fit(X_train, y_train)

joblib.dump(pipe, "models/delay_model.pkl")
pd.DataFrame({"threshold": [float(cfg["threshold"])]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))
pd.DataFrame({"column": list(medians), "median": [round(v, 4) for v in medians.values()]}) \
    .to_csv("models/impute_values.csv", index=False)

# reload check
reloaded = joblib.load("models/delay_model.pkl")
diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - pipe.predict_proba(X_test)[:, 1]).max()
pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
              "value": [len(X_test), round(float(diff), 4), float(cfg["threshold"])]}) \
    .to_csv("outputs/part_d_reload_check.csv", index=False)
print("Model packaged; reload diff", diff)
