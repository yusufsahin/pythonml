import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Veri
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 4. Görselleştir
df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
df_tsne["target"] = y

plt.figure(figsize=(8, 6))
for i, label in enumerate(target_names):
    subset = df_tsne[df_tsne["target"] == i]
    plt.scatter(subset["TSNE1"], subset["TSNE2"], label=label)
plt.title("t-SNE (Iris Dataset)")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
plt.legend()
plt.grid(True)
plt.show()
