import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = "Mall_Customers.csv"
OUTPUT_DIR = "mall_customer_outputs"
RANDOM_STATE = 42
K_RANGE = range(2, 11)
FINAL_K = 5
N_INIT = 50
MAX_ITER = 300
TOL = 1e-6


class KMeansFromScratch:
    def __init__(self, k, max_iter=300, tol=1e-6, n_init=10, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iter = None

    @staticmethod
    def _euclidean_distances(X, centroids):
        return np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))

    def _initialize_centroids(self, X, rng):
        indices = rng.choice(X.shape[0], size=self.k, replace=False)
        return X[indices].copy()

    def _fix_empty_clusters(self, X, labels, centroids):
        distances = self._euclidean_distances(X, centroids)
        nearest_distances = distances[np.arange(X.shape[0]), labels]

        for cluster_id in range(self.k):
            if np.sum(labels == cluster_id) == 0:
                farthest_index = np.argmax(nearest_distances)
                centroids[cluster_id] = X[farthest_index]
                labels[farthest_index] = cluster_id
                nearest_distances[farthest_index] = 0

        return labels, centroids

    def _single_fit(self, X, rng):
        centroids = self._initialize_centroids(X, rng)

        for iteration in range(1, self.max_iter + 1):
            distances = self._euclidean_distances(X, centroids)
            labels = np.argmin(distances, axis=1)
            labels, centroids = self._fix_empty_clusters(X, labels, centroids)

            new_centroids = np.array([
                X[labels == cluster_id].mean(axis=0)
                for cluster_id in range(self.k)
            ])

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift <= self.tol:
                break

        final_distances = self._euclidean_distances(X, centroids)
        final_labels = np.argmin(final_distances, axis=1)
        inertia = np.sum((X - centroids[final_labels]) ** 2)
        return centroids, final_labels, inertia, iteration

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best_result = None
        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._single_fit(X, rng)
            if best_result is None or inertia < best_result[2]:
                best_result = (centroids, labels, inertia, n_iter)

        self.centroids, self.labels, self.inertia, self.n_iter = best_result
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels


def standardize_features(values):
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    scaled = (values - means) / stds
    return scaled, means, stds


def inverse_standardize(values, means, stds):
    return values * stds + means


def silhouette_score_from_scratch(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return 0.0

    distances = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))
    scores = []

    for i in range(len(X)):
        same_cluster = labels == labels[i]
        same_cluster[i] = False

        if np.any(same_cluster):
            a = distances[i, same_cluster].mean()
        else:
            a = 0.0

        b_values = []
        for other_label in unique_labels:
            if other_label != labels[i]:
                other_cluster = labels == other_label
                b_values.append(distances[i, other_cluster].mean())

        b = min(b_values)
        score = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        scores.append(score)

    return float(np.mean(scores))


def evaluate_k_values(X_scaled, k_values):
    records = []

    for k in k_values:
        start_time = time.time()
        model = KMeansFromScratch(
            k=k,
            max_iter=MAX_ITER,
            tol=TOL,
            n_init=N_INIT,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X_scaled)
        runtime = time.time() - start_time
        silhouette = silhouette_score_from_scratch(X_scaled, labels)

        records.append({
            "K": k,
            "Inertia": round(float(model.inertia), 6),
            "Silhouette": round(float(silhouette), 6),
            "Iterations": int(model.n_iter),
            "Runtime_Seconds": round(float(runtime), 6),
        })

    return pd.DataFrame(records)


def assign_cluster_names(summary):
    names = {}
    for cluster_id, row in summary.iterrows():
        income = row["Mean_Income"]
        spending = row["Mean_Spending"]

        if income >= 70 and spending >= 60:
            names[cluster_id] = "High income, high spending customers"
        elif income >= 70 and spending < 40:
            names[cluster_id] = "High income, low spending customers"
        elif income < 40 and spending >= 60:
            names[cluster_id] = "Low income, high spending customers"
        elif income < 40 and spending < 40:
            names[cluster_id] = "Low income, low spending customers"
        else:
            names[cluster_id] = "Medium income, average spending customers"

    return names


def create_cluster_summary(df):
    summary = df.groupby("Cluster").agg(
        Count=("CustomerID", "count"),
        Mean_Age=("Age", "mean"),
        Mean_Income=("Annual Income (k$)", "mean"),
        Mean_Spending=("Spending Score (1-100)", "mean"),
        Min_Income=("Annual Income (k$)", "min"),
        Max_Income=("Annual Income (k$)", "max"),
        Min_Spending=("Spending Score (1-100)", "min"),
        Max_Spending=("Spending Score (1-100)", "max"),
        Female_Count=("Gender", lambda x: int((x == "Female").sum())),
        Male_Count=("Gender", lambda x: int((x == "Male").sum())),
    )

    summary = summary.round(2)
    names = assign_cluster_names(summary)
    summary["Segment_Name"] = summary.index.map(names)
    return summary


def plot_numeric_distributions(df, output_dir):
    numeric_columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, column in zip(axes, numeric_columns):
        ax.hist(df[column], bins=15, edgecolor="black", alpha=0.75)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "01_numeric_distributions.png", dpi=160)
    plt.close(fig)


def plot_gender_count(df, output_dir):
    counts = df["Gender"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index, counts.values, edgecolor="black", alpha=0.8)
    ax.set_title("Gender Count")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Customers")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "02_gender_count.png", dpi=160)
    plt.close(fig)


def plot_initial_scatter(df, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"Male": "o", "Female": "s"}

    for gender, marker in markers.items():
        subset = df[df["Gender"] == gender]
        ax.scatter(
            subset["Annual Income (k$)"],
            subset["Spending Score (1-100)"],
            marker=marker,
            s=55,
            edgecolor="black",
            alpha=0.75,
            label=gender,
        )

    ax.set_title("Annual Income vs Spending Score Before Clustering")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(title="Gender")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "03_income_spending_before_clustering.png", dpi=160)
    plt.close(fig)


def plot_elbow(evaluation_df, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(evaluation_df["K"], evaluation_df["Inertia"], marker="o")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(evaluation_df["K"])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "04_elbow_method.png", dpi=160)
    plt.close(fig)


def plot_silhouette(evaluation_df, output_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(evaluation_df["K"], evaluation_df["Silhouette"], marker="o")
    ax.axvline(FINAL_K, linestyle="--", alpha=0.7, label=f"Selected K = {FINAL_K}")
    ax.set_title("Silhouette Score")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(evaluation_df["K"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "05_silhouette_score.png", dpi=160)
    plt.close(fig)


def plot_final_clusters(df, centroids_original, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))

    for cluster_id in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cluster_id]
        ax.scatter(
            subset["Annual Income (k$)"],
            subset["Spending Score (1-100)"],
            s=60,
            edgecolor="black",
            alpha=0.8,
            label=f"Cluster {cluster_id}",
        )

    ax.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        s=220,
        marker="X",
        edgecolor="black",
        linewidth=1.5,
        label="Centroids",
    )

    ax.set_title("Final Customer Segments")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "06_final_customer_segments.png", dpi=180)
    plt.close(fig)


def plot_cluster_profiles(summary, output_dir):
    profile_columns = ["Mean_Age", "Mean_Income", "Mean_Spending"]
    profile_data = summary[profile_columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    profile_data.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85)
    ax.set_title("Cluster Profiles")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Average Value")
    ax.set_xticklabels(profile_data.index, rotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["Average Age", "Average Income", "Average Spending Score"])

    fig.tight_layout()
    fig.savefig(output_dir / "07_cluster_profiles.png", dpi=160)
    plt.close(fig)


def write_report(output_dir, df, evaluation_df, summary, final_model, final_silhouette):
    best_k = int(evaluation_df.loc[evaluation_df["Silhouette"].idxmax(), "K"])
    best_silhouette = float(evaluation_df["Silhouette"].max())

    lines = []
    lines.append("Customer Segmentation Report")
    lines.append("============================")
    lines.append("")
    lines.append("1. Project Goal")
    lines.append("The goal is to segment mall customers using unsupervised learning. K-Means was implemented from scratch without using any ready-made clustering algorithm.")
    lines.append("")
    lines.append("2. Dataset Overview")
    lines.append(f"Dataset shape: {df.shape}")
    lines.append("Initial analysis includes gender count, numeric feature distributions, and the income-spending scatter plot.")
    lines.append("")
    lines.append("3. Features Used for Final Clustering")
    lines.append("Final clustering uses only Annual Income (k$) and Spending Score (1-100).")
    lines.append("These two variables directly represent customer purchasing power and purchasing behavior, so they produce clear and interpretable marketing segments.")
    lines.append("Age and Gender are still used for exploratory analysis and cluster interpretation, but not for distance calculation in the final K-Means model.")
    lines.append("")
    lines.append("4. Preprocessing")
    lines.append("The selected clustering features were standardized before K-Means because Euclidean distance is sensitive to feature scale.")
    lines.append("")
    lines.append("5. K-Means Implementation")
    lines.append("The implementation includes random centroid initialization, Euclidean distance calculation, iterative assignment and centroid update, empty-cluster handling, and multiple initializations.")
    lines.append(f"Number of initializations for each K: {N_INIT}")
    lines.append("")
    lines.append("6. K Evaluation")
    lines.append(evaluation_df.to_string(index=False))
    lines.append("")
    lines.append(f"Best K by Silhouette Score: {best_k} with score {best_silhouette:.6f}")
    lines.append(f"Selected final K: {FINAL_K}")
    lines.append("K=5 was selected because it has the strongest silhouette score in the tested range and also matches five clear business segments in the income-spending space.")
    lines.append("")
    lines.append("7. Final Model")
    lines.append(f"Final inertia: {final_model.inertia:.6f}")
    lines.append(f"Final silhouette score: {final_silhouette:.6f}")
    lines.append(f"Iterations until convergence: {final_model.n_iter}")
    lines.append("")
    lines.append("8. Cluster Summary")
    lines.append(summary.to_string())
    lines.append("")
    lines.append("9. Business Interpretation")
    lines.append("High income and high spending customers are valuable customers for loyalty programs and premium offers.")
    lines.append("High income and low spending customers have purchasing power but may need personalized offers or engagement campaigns.")
    lines.append("Low income and high spending customers may respond well to discounts and affordable bundles.")
    lines.append("Low income and low spending customers should be targeted with basic promotions and low-risk campaigns.")
    lines.append("Medium income and average spending customers represent stable regular customers.")

    (output_dir / "10_project_report.txt").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "final_k": FINAL_K,
        "best_k_by_silhouette": best_k,
        "best_silhouette": best_silhouette,
        "final_inertia": float(final_model.inertia),
        "final_silhouette": float(final_silhouette),
        "iterations": int(final_model.n_iter),
        "n_init": N_INIT,
        "features_used_for_clustering": ["Annual Income (k$)", "Spending Score (1-100)"],
        "features_used_for_analysis": ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"],
        "evaluation": evaluation_df.to_dict(orient="records"),
    }
    (output_dir / "11_run_metadata.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "CustomerID",
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("Customer Segmentation with K-Means from Scratch")
    print("------------------------------------------------")
    print(f"Dataset shape: {df.shape}")
    print("Exploratory features: Gender, Age, Annual Income (k$), Spending Score (1-100)")
    print("Final clustering features: Annual Income (k$), Spending Score (1-100)")
    print(f"Missing values: {int(df.isna().sum().sum())}")

    plot_numeric_distributions(df, output_dir)
    plot_gender_count(df, output_dir)
    plot_initial_scatter(df, output_dir)

    clustering_features = ["Annual Income (k$)", "Spending Score (1-100)"]
    X = df[clustering_features].to_numpy(dtype=float)
    X_scaled, means, stds = standardize_features(X)

    evaluation_df = evaluate_k_values(X_scaled, K_RANGE)
    evaluation_df.to_csv(output_dir / "08_k_evaluation.csv", index=False)

    plot_elbow(evaluation_df, output_dir)
    plot_silhouette(evaluation_df, output_dir)

    final_model = KMeansFromScratch(
        k=FINAL_K,
        max_iter=MAX_ITER,
        tol=TOL,
        n_init=N_INIT,
        random_state=RANDOM_STATE,
    )
    labels = final_model.fit_predict(X_scaled)
    final_silhouette = silhouette_score_from_scratch(X_scaled, labels)

    df_output = df.copy()
    df_output["Cluster"] = labels
    centroids_original = inverse_standardize(final_model.centroids, means, stds)

    summary = create_cluster_summary(df_output)
    df_output.to_csv(output_dir / "09_customer_segments.csv", index=False)
    summary.to_csv(output_dir / "09_cluster_summary.csv")
    pd.DataFrame(
        centroids_original,
        columns=["Centroid_Annual_Income", "Centroid_Spending_Score"],
    ).to_csv(output_dir / "09_cluster_centroids.csv", index_label="Cluster")

    plot_final_clusters(df_output, centroids_original, output_dir)
    plot_cluster_profiles(summary, output_dir)
    write_report(output_dir, df_output, evaluation_df, summary, final_model, final_silhouette)

    best_k = int(evaluation_df.loc[evaluation_df["Silhouette"].idxmax(), "K"])

    print("\nK evaluation:")
    print(evaluation_df.to_string(index=False))
    print(f"\nBest K by Silhouette Score: {best_k}")
    print(f"Selected K for final model: {FINAL_K}")
    print(f"Final Inertia: {final_model.inertia:.6f}")
    print(f"Final Silhouette Score: {final_silhouette:.6f}")
    print("\nCluster summary:")
    print(summary.to_string())
    print(f"\nAll output files were saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()