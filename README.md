# BSc Computational Intelligence — Course Projects

Programming assignments from the **Computational Intelligence** course (BSc Computer Engineering,
University of Isfahan). The algorithms are implemented from scratch with NumPy instead of calling ready-made
library models, and then evaluated and interpreted.

## Contents

| Folder | Topic | What is implemented | Key result |
|---|---|---|---|
| [`ci1-kmeans-customer-segmentation`](ci1-kmeans-customer-segmentation) | Unsupervised learning | K-Means, the silhouette score and standardization written from scratch; elbow and silhouette analysis; marketing segments of 200 mall customers | Five customer segments, silhouette 0.555 |

Each project folder has its own README with the task, the approach, the results and how to run it.

```
BSc-Computational-Intelligence/
├── ci1-kmeans-customer-segmentation/
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
