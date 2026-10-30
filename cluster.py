"""Group keywords by meaning using agglomerative clustering on their embeddings."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

OUTPUT_DIR = Path("output")
# These keywords sit close together, so the useful thresholds are small. Sweep in fine steps.
SWEEP = [round(t, 2) for t in np.arange(0.01, 0.151, 0.01)]
TARGET_CLUSTER_COUNTS = [8, 15, 24, 40]


def cluster(vectors, threshold):
    # Start with every keyword as its own cluster, then keep merging the closest pair.
    # "Closest" = average cosine distance between all members of the two clusters.
    # Stop merging once the closest pair is farther apart than the threshold.
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    return model.fit_predict(vectors)


def sweep(vectors):
    # Try a range of thresholds to see how the number of clusters changes.
    # Lower threshold = stricter = more, smaller clusters.
    rows = []
    for t in SWEEP:
        labels = cluster(vectors, t)
        sizes = np.bincount(labels)
        rows.append(
            {
                "threshold": t,
                "clusters": len(sizes),
                "singletons": int((sizes == 1).sum()),
                "largest": int(sizes.max()),
            }
        )
    table = pd.DataFrame(rows)
    print("\nThreshold sweep")
    print(table.to_string(index=False))
    return table


def pick_thresholds(table):
    # For each target cluster count, pick the sweep threshold that lands closest to it.
    picked = {}
    for target in TARGET_CLUSTER_COUNTS:
        row = table.iloc[(table["clusters"] - target).abs().argmin()]
        picked.setdefault(row["threshold"], []).append(target)
        print(f"Target {target:>2} clusters -> threshold {row['threshold']:.2f} ({int(row['clusters'])} clusters)")

    # If two targets land on the same threshold, show it once and say so.
    for t, targets in picked.items():
        if len(targets) > 1:
            print(f"Note: targets {targets} all landed on threshold {t:.2f}, printing it once")
    return sorted(picked)


def show_and_save(df, vectors, threshold):
    out = df[["planner_rank", "keyword"]].copy()
    out["cluster_id"] = cluster(vectors, threshold)

    # Print clusters biggest first, keywords inside each in planner_rank order.
    groups = sorted(out.groupby("cluster_id"), key=lambda g: (-len(g[1]), g[1]["planner_rank"].min()))
    print(f"\n{'=' * 60}\nThreshold {threshold:.2f}: {len(groups)} clusters\n{'=' * 60}")
    for cid, g in groups:
        print(f"\nCluster {cid} ({len(g)} keywords)")
        for _, row in g.sort_values("planner_rank").iterrows():
            print(f"  {row.planner_rank:>3}  {row.keyword}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"clusters_t{threshold:.2f}.csv"
    out.sort_values("planner_rank").to_csv(path, index=False)
    print(f"\nSaved {path}")


def run(df, vectors, thresholds=None):
    table = sweep(vectors)
    # Use the thresholds passed in, or default to the ones nearest 8 / 15 / 24 / 40 clusters.
    thresholds = thresholds or pick_thresholds(table)
    print(f"\nShowing thresholds: {', '.join(f'{t:.2f}' for t in thresholds)}")
    for t in thresholds:
        show_and_save(df, vectors, t)
