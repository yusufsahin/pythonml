import pandas as pd
import umap
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Veri seti (yüksek boyutlu)
data = load_digits()
X = data.data
y = data.target

# Ölçekleme
X_scaled = StandardScaler().fit_transform(X)

# UMAP ile boyut indirgeme
umap_model = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    metric='euclidean',
    random_state=42
)
X_umap = umap_model.fit_transform(X_scaled)

# Görselleştirme
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=10)
plt.colorbar(scatter)
plt.title("UMAP Projection of Digits Dataset", fontsize=14)
plt.show()
