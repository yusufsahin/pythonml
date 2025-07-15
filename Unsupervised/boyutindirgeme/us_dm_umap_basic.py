import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt

# 1. Veri seti
iris = load_iris()
X = iris.data
y = iris.target

# 2. Normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. UMAP ile boyut indir
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# 4. Sonuçları görselleştir
df = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])
df["Target"] = y

plt.figure(figsize=(8, 6))
for i in range(3):
    subset = df[df["Target"] == i]
    plt.scatter(subset["UMAP1"], subset["UMAP2"], label=f"Sınıf {i}")
plt.title("UMAP - Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
