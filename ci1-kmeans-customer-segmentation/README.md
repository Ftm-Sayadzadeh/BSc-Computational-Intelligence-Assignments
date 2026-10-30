# Customer Segmentation with K-Means — Implemented from Scratch

Group the customers of a shopping mall into marketing segments with **K-Means clustering written from scratch**
(NumPy only — no `sklearn.cluster`, and not even `sklearn` for scaling or for the silhouette score), then choose the
number of clusters with the elbow method and the silhouette score and interpret the segments.

## Data

`Mall_Customers.csv` — 200 customers with `CustomerID`, `Gender` (112 female / 88 male), `Age`,
`Annual Income (k$)` and `Spending Score (1-100)`. There are no missing values. This is the Mall Customer
Segmentation dataset (please check its original source for the license and citation).

## Approach

1. **Exploration.** Histograms of age, income and spending score, the gender counts, and a scatter plot of income against spending score before clustering.
2. **Features for clustering.** Only `Annual Income` and `Spending Score` are used to compute distances: together they describe purchasing power and purchasing behaviour and give segments that can be read directly in 2D. `Age` and `Gender` are kept for describing the clusters afterwards.
3. **Standardization.** Both features are converted to z-scores (written by hand), because Euclidean distance is sensitive to scale.
4. **`KMeansFromScratch`:**
   - random initialization (K distinct customers as the first centroids),
   - vectorized Euclidean distances,
   - repeated assignment to the nearest centroid and centroid update until the centroids move less than `1e-6` (at most 300 iterations),
   - empty-cluster handling (an empty cluster takes the point farthest from its centroid),
   - **50 random restarts** for every K, keeping the run with the lowest inertia.
5. **Choosing K.** For K = 2 … 10 the script records the inertia (elbow method) and the **silhouette score**, which is also implemented from scratch.
6. **Final model and interpretation.** K = 5; each cluster is described by its size, mean age, income and spending, gender split, and named from its income and spending level.

## Results

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 269.02 | 0.295 |
| 3 | 157.70 | 0.467 |
| 4 | 108.92 | 0.494 |
| **5** | **65.57** | **0.555** |
| 6 | 55.06 | 0.540 |
| 7 | 44.86 | 0.528 |
| 8 | 37.15 | 0.456 |
| 9 | 32.38 | 0.459 |
| 10 | 29.09 | 0.452 |

(Inertia is measured on the standardized features.)

- **K = 5 is supported by both criteria.** The inertia falls by 43.4 going from 4 to 5 clusters but only by 10.5 from 5 to 6 — a clear elbow — and the silhouette score peaks at K = 5 (0.555). K = 6 and 7 are close (0.540 and 0.528), so the choice is clear but not razor-sharp; a silhouette around 0.55 means well-separated, sensible groups rather than perfectly separated ones.
- The final model converged in 10 iterations.

### The five segments

| Cluster | Customers | Mean age | Mean income (k$) | Mean spending | Segment |
|---|---|---|---|---|---|
| 0 | 39 | 32.7 | 86.5 | 82.1 | High income, high spending |
| 1 | 23 | 45.2 | 26.3 | 20.9 | Low income, low spending |
| 2 | 22 | 25.3 | 25.7 | 79.4 | Low income, high spending |
| 3 | 35 | 41.1 | 88.2 | 17.1 | High income, low spending |
| 4 | 81 | 42.7 | 55.3 | 49.5 | Medium income, average spending |

- **Cluster 4** is the largest group (40 % of customers): ordinary shoppers with medium income and average spending.
- **Clusters 0 and 2** are the high spenders and are also the youngest (mean age 33 and 25); the low spenders (clusters 1 and 3) are older (45 and 41). This is a pattern in the cluster means, not something the model used.
- **Cluster 3** (high income, low spending) is the group with unused purchasing power, and the natural target for personalised offers; **cluster 0** is the most valuable group for loyalty programmes; **cluster 2** is likely to respond to discounts and bundles.
- The segment names come from simple thresholds on each cluster's mean income and spending (`assign_cluster_names` in the script), and gender is split fairly evenly in every cluster.

## Output files (`mall_customer_outputs/`)

| File | Content |
|---|---|
| `01_numeric_distributions.png`, `02_gender_count.png`, `03_income_spending_before_clustering.png` | Exploratory plots |
| `04_elbow_method.png`, `05_silhouette_score.png` | Choosing K |
| `06_final_customer_segments.png` | The five clusters and their centroids |
| `07_cluster_profiles.png` | Average age, income and spending per cluster |
| `08_k_evaluation.csv` | Inertia, silhouette, iterations and run time for every K |
| `09_cluster_summary.csv`, `09_cluster_centroids.csv`, `09_customer_segments.csv` | Cluster statistics, centroids (original units) and the cluster of every customer |
| `10_project_report.txt`, `11_run_metadata.json` | Text report and run settings |

## Run

```bash
pip install -r ../requirements.txt
python customer_segmentation_kmeans.py
```

The settings are constants at the top of the script (`K_RANGE`, `FINAL_K`, `N_INIT`, `RANDOM_STATE`, …). The random
seed is fixed (42), so a re-run reproduces the K evaluation, the clusters and all tables exactly (only the run times differ).
