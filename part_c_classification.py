"""Part C: delay classifier, feature scoring, tuning and alert threshold."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"]

# 1. stratified split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
print("sizes", len(X_train), len(X_val), len(X_test), "delayed", y_train.sum(), y_val.sum(), y_test.sum())


def cls_row(name, y_true, y_pred, **extra):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {**extra, "model": name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


# 2. baseline + three classifiers (default 0.5 rule)
rows = [cls_row("blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int))]
lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]).fit(X_train, y_train)
rows.append(cls_row("logistic_regression", y_val, lr.predict(X_val)))
best_d, best_f1 = None, -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
    f = f1_score(y_val, t.predict(X_val))
    if f > best_f1:                 # strict > keeps the smaller depth on ties
        best_d, best_f1 = d, f
tree = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
rows.append(cls_row(f"decision_tree_depth_{best_d}", y_val, tree.predict(X_val)))
rf0 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
rows.append(cls_row("random_forest_default", y_val, rf0.predict(X_val)))
pd.DataFrame(rows).round(4).to_csv("outputs/part_c_models.csv", index=False)

# 3. feature signal
sel_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
fs = pd.DataFrame({"feature": FEATURES, "f_score": sel_all.scores_, "rf_importance": rf0.feature_importances_})
fs["f_rank"] = fs["f_score"].rank(ascending=False, method="first").astype(int)
fs["is_noise"] = fs["feature"].isin(NOISE_COLUMNS)
fs = fs[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
fs[["f_score", "rf_importance"]] = fs[["f_score", "rf_importance"]].round(4)
fs.to_csv("outputs/part_c_feature_scores.csv", index=False)

sk_rows = []
for k in [5, 7, 10, 12, 17]:
    p = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                  ("model", RandomForestClassifier(n_estimators=100, random_state=42))]).fit(X_train, y_train)
    sk_rows.append({"k": k, "val_f1": f1_score(y_val, p.predict(X_val))})
pd.DataFrame(sk_rows).round(4).to_csv("outputs/part_c_selectk.csv", index=False)

# 4. tuning on 7 selected features (selector fitted on training rows only)
K = 7
selector = SelectKBest(f_classif, k=K).fit(X_train, y_train)
Xtr7, Xva7 = selector.transform(X_train), selector.transform(X_val)
grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, cv=5).fit(Xtr7, y_train)
rs = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=9, cv=5, random_state=42).fit(Xtr7, y_train)
default7 = RandomForestClassifier(random_state=42).fit(Xtr7, y_train)
tune = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 27 * 5,
     "best_params": dict(gs.best_params_), "best_cv_accuracy": gs.best_score_,
     "val_f1": f1_score(y_val, gs.best_estimator_.predict(Xva7))},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 9 * 5,
     "best_params": dict(rs.best_params_), "best_cv_accuracy": rs.best_score_,
     "val_f1": f1_score(y_val, rs.best_estimator_.predict(Xva7))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}, "best_cv_accuracy": np.nan,
     "val_f1": f1_score(y_val, default7.predict(Xva7))},
]
tdf = pd.DataFrame(tune)
tdf[["best_cv_accuracy", "val_f1"]] = tdf[["best_cv_accuracy", "val_f1"]].round(4)
tdf.to_csv("outputs/part_c_tuning.csv", index=False)

# 5. final configuration: highest val_f1, ties -> default, grid, randomized
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]
cand = {r["search"]: r for r in tune}
top = max(r["val_f1"] for r in tune)
winner = next(n for n in priority if cand[n]["val_f1"] == top)
params = cand[winner]["best_params"]
final = Pipeline([("selector", SelectKBest(f_classif, k=K)),
                  ("model", RandomForestClassifier(random_state=42, **params))]).fit(X_train, y_train)

# 6. alert threshold on validation
proba_val = final.predict_proba(X_val)[:, 1]
th_rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    r = cls_row("x", y_val, (proba_val >= t).astype(int)); r.pop("model"); th_rows.append({"threshold": t, **r})
thdf = pd.DataFrame(th_rows)
ok = thdf[thdf["recall"] >= 0.80]
threshold = float(ok["threshold"].max()) if len(ok) else 0.1
thdf.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)

# 7. single test evaluation
proba_test = final.predict_proba(X_test)[:, 1]
r = cls_row("x", y_test, (proba_test >= threshold).astype(int)); r.pop("model")
pd.DataFrame([{"threshold": threshold, **r}]).round(4).to_csv("outputs/part_c_final_test.csv", index=False)
pd.DataFrame([{"selector_k": K, "n_estimators": params["n_estimators"],
               "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
               "min_samples_leaf": params["min_samples_leaf"], "threshold": threshold,
               "chosen_from": winner}]).to_csv("outputs/part_c_final_config.csv", index=False)
print("Part C done; chosen:", winner, params, "threshold", threshold)
