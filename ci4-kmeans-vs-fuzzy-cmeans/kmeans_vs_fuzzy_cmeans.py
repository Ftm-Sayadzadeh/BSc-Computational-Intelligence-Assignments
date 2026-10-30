import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# =========================
# Global Seed (Reproducibility)
# =========================
np.random.seed(42)

# =========================
# Load Dataset
# =========================
iris = load_iris()
X = iris.data


# =========================
# K-Means (Scratch)
# =========================
class KMeansScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        n = X.shape[0]
        idx = np.random.choice(n, self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, None] - self.centroids[None, :], axis=2)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[self.labels == j].mean(axis=0) if np.any(self.labels == j)
                else self.centroids[j]
                for j in range(self.k)
            ])

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        return self

    def inertia(self, X):
        return np.sum([
            np.sum((X[i] - self.centroids[self.labels[i]]) ** 2)
            for i in range(X.shape[0])
        ])


# =========================
# Fuzzy C-Means (Scratch)
# =========================
class FuzzyCMeansScratch:
    def __init__(self, c=3, m=2, max_iters=100, tol=1e-5, seed=42):
        self.c = c
        self.m = m
        self.max_iters = max_iters
        self.tol = tol
        np.random.seed(seed)  

    def initialize_U(self, n):
        U = np.random.rand(n, self.c)
        return U / np.sum(U, axis=1, keepdims=True)

    def compute_centroids(self, X, U):
        um = U ** self.m
        return (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

    def update_U(self, X, centroids):
        n = X.shape[0]
        U = np.zeros((n, self.c))
        dist = np.linalg.norm(X[:, None] - centroids[None, :], axis=2) + 1e-8

        for i in range(n):
            for j in range(self.c):
                U[i, j] = 1 / np.sum(
                    (dist[i, j] / dist[i]) ** (2 / (self.m - 1))
                )

        return U

    def fit(self, X):
        n = X.shape[0]
        U = self.initialize_U(n)

        for _ in range(self.max_iters):
            U_old = U.copy()

            centroids = self.compute_centroids(X, U)
            U = self.update_U(X, centroids)

            if np.linalg.norm(U - U_old) < self.tol:
                break

        self.U = U
        self.centroids = centroids
        return self

    # =========================
    # CLEAN FUZZY INERTIA
    # =========================
    def inertia(self, X):
        um = self.U ** self.m
        inertia = 0.0

        for j in range(self.c):
            diff = X - self.centroids[j]
            inertia += np.sum(um[:, j][:, None] * (diff ** 2))

        return inertia

    def predict(self):
        return np.argmax(self.U, axis=1)


# =========================
# Visualization
# =========================
def plot_and_save(X, labels, centroids, title, filename):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200)
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# MAIN CLEAN EXECUTION
# =========================
def main():
    kmeans = KMeansScratch(k=3).fit(X)
    fcm = FuzzyCMeansScratch(c=3, seed=42).fit(X)

    k_inertia = kmeans.inertia(X)
    f_inertia = fcm.inertia(X)

    print("KMeans Inertia:", k_inertia)
    print("Fuzzy CMeans Inertia:", f_inertia)

    # Sepal
    plot_and_save(X[:, :2], kmeans.labels, kmeans.centroids,
                  "KMeans - Sepal", "kmeans_sepal.png")

    plot_and_save(X[:, :2], fcm.predict(), fcm.centroids,
                  "Fuzzy - Sepal", "fuzzy_sepal.png")

    # Petal
    plot_and_save(X[:, 2:], kmeans.labels, kmeans.centroids[:, 2:],
                  "KMeans - Petal", "kmeans_petal.png")

    plot_and_save(X[:, 2:], fcm.predict(), fcm.centroids[:, 2:],
                  "Fuzzy - Petal", "fuzzy_petal.png")


if __name__ == "__main__":
    main()