# BSc Computational Intelligence — Course Projects

Programming assignments from the **Computational Intelligence** course (BSc Computer Engineering,
University of Isfahan). The algorithms are implemented from scratch with NumPy instead of calling ready-made
library models, and then evaluated and interpreted.

## Contents

| Folder | Topic | What is implemented | Key result |
|---|---|---|---|
| [`ci1-kmeans-customer-segmentation`](ci1-kmeans-customer-segmentation) | Unsupervised learning | K-Means, the silhouette score and standardization written from scratch; elbow and silhouette analysis; marketing segments of 200 mall customers | Five customer segments, silhouette 0.555 |
| [`ci2-mnist-neural-network`](ci2-mnist-neural-network) | Neural networks | Fully connected network, backpropagation, ReLU/sigmoid/tanh, softmax + cross-entropy, mini-batch GD, Adam, dropout, early stopping and saved weights — all from scratch with NumPy | Test accuracy 97.89 % on MNIST |
| [`ci3-genetic-algorithm-tsp`](ci3-genetic-algorithm-tsp) | Evolutionary optimization | A genetic algorithm for the Traveling Salesperson Problem: permutation chromosomes, tournament selection, Order Crossover, inversion mutation, elitism; convergence plots and route animations for 20, 50 and 100 cities | Valid tours within about 1–5 % of a 2-opt reference |
| [`ci4-kmeans-vs-fuzzy-cmeans`](ci4-kmeans-vs-fuzzy-cmeans) | Fuzzy clustering | K-Means and Fuzzy C-Means written from scratch and compared on Iris; hard versus soft clustering | 148 of 150 points get the same cluster; 13 points are genuinely ambiguous |

Each project folder has its own README with the task, the approach, the results and how to run it.

```
BSc-Computational-Intelligence/
├── ci1-kmeans-customer-segmentation/
├── ci2-mnist-neural-network/
├── ci3-genetic-algorithm-tsp/
├── ci4-kmeans-vs-fuzzy-cmeans/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Every script reads its data and writes its results relative to the **current directory**, so run it from inside its
own project folder:

```bash
cd ci1-kmeans-customer-segmentation
python customer_segmentation_kmeans.py
```

## Author

Fatemeh Sayadzadeh — [GitHub](https://github.com/Ftm-Sayadzadeh)
