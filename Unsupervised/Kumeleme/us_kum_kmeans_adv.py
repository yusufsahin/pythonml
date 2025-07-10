import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# === 1. VERİ YÜKLE ===
df = pd.read_csv("data/Mall_Customers.csv")

# === 2. GEREKSİZ KOLONLARI AT (CustomerID) ===
df.drop("CustomerID", axis=1, inplace=True)

# === 3. Cinsiyeti Etiketle ===
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])  # Male=1, Female=0

# === 4. VERİYİ ÖLÇEKLE ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# === 5. PCA İLE 2D BOYUT İNDİRME ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# === 6. ELBOW METHOD ===
wcss = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K, wcss, 'bo-')
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("WCSS (inertia)")
plt.title("Elbow Yöntemi ile Optimum K")
plt.grid()
plt.tight_layout()
plt.show()

# === 7. KMeans Uygula (k=5) ===
kmeans_final = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans_final.fit_predict(X_scaled)

# === 8. Silhouette Score ===
sil_score = silhouette_score(X_scaled, df["Cluster"])
print(f"Silhouette Score (k=5): {sil_score:.4f}")

# === 9. GÖRSELLEŞTİRME ===
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set1", s=80)
plt.title("K-Means (PCA ile Görselleştirme)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 10. Küme Ortalamaları ===
print("\nCluster Ortalamaları:")
print(df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean())
