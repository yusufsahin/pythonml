import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap  # pip install umap-learn

# 1. Veri seti: Rakam görselleri
digits = load_digits()
X = digits.data
y = digits.target

# 2. Normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 5. UMAP
umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# 6. Görselleştir
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="Spectral", s=10)
axes[0].set_title("PCA")

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="Spectral", s=10)
axes[1].set_title("t-SNE")

axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="Spectral", s=10)
axes[2].set_title("UMAP")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Advanced Comparison: PCA vs t-SNE vs UMAP on Digits Dataset", fontsize=14)
plt.tight_layout()
plt.show()
