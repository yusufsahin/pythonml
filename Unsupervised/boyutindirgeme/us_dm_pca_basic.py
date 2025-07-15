import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Iris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA ile 2 bileşene indir
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. DataFrame'e dönüştür
df_pca = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
df_pca["target"] = y

# 5. Görselleştir
colors = ["red", "green", "blue"]
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(
        df_pca[df_pca["target"] == i]["PC1"],
        df_pca[df_pca["target"] == i]["PC2"],
        label=target_name,
        color=colors[i]
    )
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
