"""Part D: hub profile, K-Means segments, separation evidence, tier-vs-segment check."""
import os
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
hubs = pd.read_csv("data/hubs.csv")

# 1. hub profile (ascending hub_id, as groupby returns it)
profile = df.groupby("hub_id").agg(shipments=("shipment_id", "count"),
                                   mean_delivery_hours=("delivery_hours", "mean"),
                                   delay_rate=("is_delayed", "mean"),
                                   mean_distance_km=("distance_km", "mean"),
                                   mean_n_stops=("n_stops", "mean")).reset_index().sort_values("hub_id")
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)
cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]

# 2. scale + K-Means
Z = StandardScaler().fit_transform(profile[cols])
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Z)
profile["cluster"] = km.labels_

# 3. separation evidence for K = 2..5
D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
sep = []
for k in [2, 3, 4, 5]:
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z).labels_
    same, diff = [], []
    for i, j in combinations(range(len(Z)), 2):
        (same if lab[i] == lab[j] else diff).append(D[i, j])
    sizes = sorted(np.bincount(lab), reverse=True)
    sep.append({"K": k, "across_over_within": np.mean(diff) / np.mean(same),
                "cluster_sizes": "-".join(str(s) for s in sizes)})
pd.DataFrame(sep).round(4).to_csv("outputs/part_d_separation.csv", index=False)

# 4. name segments by mean delay rate of their hubs
order = profile.groupby("cluster")["delay_rate"].mean().sort_values().index.tolist()
names = dict(zip(order, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(names)
seg = profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")[["hub_id", "segment", "hub_city", "hub_tier"]]
seg.to_csv("outputs/hub_segments.csv", index=False)

sp = profile.groupby("segment").agg(hubs=("hub_id", "count"), mean_delivery_hours=("mean_delivery_hours", "mean"),
                                    delay_rate=("delay_rate", "mean"), mean_distance_km=("mean_distance_km", "mean"),
                                    mean_n_stops=("mean_n_stops", "mean")).reindex(["Standard", "Watch", "Escalate"])
sp.reset_index().round(4).to_csv("outputs/segment_profile.csv", index=False)

implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
seg[seg["segment"] != seg["hub_tier"].map(implied)].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print("Part D segments done")
