import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# === 1. VERİ SETİ OLUŞTUR (Yapay Müşteri Verisi) ===
np.random.seed(42)
data = {
    "Age": np.random.randint(18, 70, 200),
    "Annual Income (k$)": np.random.randint(20, 120, 200),
    "Spending Score (1-100)": np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)

# === 2. BASIC K-MEANS ===
X_basic = df[["Annual Income (k$)", "Spending Score (1-100)"]]
kmeans_basic = KMeans(n_clusters=5, random_state=42)
df["Cluster_Basic"] = kmeans_basic.fit_predict(X_basic)

# === 3. ADVANCED K-MEANS ===
# a) Tüm özellikleri ölçekle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])

# b) PCA ile boyut indir
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# c) K-Means clustering (Advanced)
kmeans_adv = KMeans(n_clusters=5, random_state=42)
df["Cluster_Advanced"] = kmeans_adv.fit_predict(X_scaled)

# d) Silhouette score hesapla
sil_score = silhouette_score(X_scaled, df["Cluster_Advanced"])
print(f"Silhouette Score (Advanced): {sil_score:.4f}")

# === 4. GÖRSELLEŞTİRME ===
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster_Advanced", palette="tab10")
plt.title("Advanced K-Means (PCA ile)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 5. İsteğe bağlı: Cluster temel istatistikleri
print("\nCluster Ortalamaları (Advanced):")
print(df.groupby("Cluster_Advanced")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean())
