
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# === 1. VERİYİ YÜKLE ===
df = pd.read_csv("data/Mall_Customers.csv")
df.drop("CustomerID", axis=1, inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

# === 2. SADECE GELİR & HAR-CAMA KULLAN (BASIC) ===
X_basic = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_basic_scaled = StandardScaler().fit_transform(X_basic)

# === 3. K-DISTANCE GRAFİĞİ (eps belirleme için) ===
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_basic_scaled)
distances, indices = neighbors_fit.kneighbors(X_basic_scaled)
distances = np.sort(distances[:, 4])

plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.title("K-Distance Graph (DBSCAN için eps belirleme)")
plt.xlabel("Veri Noktası")
plt.ylabel("Uzaklık (5. komşu)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 4. DBSCAN UYGULA (eps=0.5 varsayalım) ===
dbscan = DBSCAN(eps=0.5, min_samples=5)
df["Cluster_DBSCAN"] = dbscan.fit_predict(X_basic_scaled)

# === 5. KMEANS KARŞILAŞTIRMA (2D) ===
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster_KMeans"] = kmeans.fit_predict(X_basic_scaled)

# === 6. PCA İLE ADVANCED DBSCAN (Tüm verilerle) ===
X_all = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]]
X_all_scaled = StandardScaler().fit_transform(X_all)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

dbscan_adv = DBSCAN(eps=0.5, min_samples=5)
df["Cluster_DBSCAN_Adv"] = dbscan_adv.fit_predict(X_all_scaled)

# === 7. SİLUET SKORU (Opsiyonel)
print(f"Silhouette Score (KMeans): {silhouette_score(X_basic_scaled, df['Cluster_KMeans']):.4f}")
if len(set(df['Cluster_DBSCAN_Adv'])) > 1:
    print(f"Silhouette Score (DBSCAN Adv): {silhouette_score(X_all_scaled, df['Cluster_DBSCAN_Adv']):.4f}")
else:
    print("DBSCAN Advanced küme sayısı yetersiz (tek sınıf çıktı).")

# === 8. GÖRSELLER ===
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster_KMeans", palette="Set1")
plt.title("K-Means (Basic 2D)")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster_DBSCAN", palette="Set2")
plt.title("DBSCAN (Basic 2D)")

plt.tight_layout()
plt.show()

# PCA üzerinden DBSCAN Advanced görselleştirme
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster_DBSCAN_Adv", palette="Set2")
plt.title("DBSCAN (Advanced PCA Görselleştirme)")
plt.grid(True)
plt.tight_layout()
plt.show()
