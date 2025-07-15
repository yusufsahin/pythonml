import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veri setini oku
df = pd.read_csv("data/Mall_Customers.csv")

# 2. Kullanılacak özellikler (yaş, gelir, harcama)
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# 3. Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. GMM ile kümeleme
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)

# 5. Sonuçları dataframe'e ekle
df["GMM_Cluster"] = labels

# 6. Kümeleme sonuçlarını analiz et
print(df.groupby("GMM_Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean())

# 7. PCA ile 2D görselleştirme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:,0]
df["PCA2"] = X_pca[:,1]

# 8. Görselleştirme
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="GMM_Cluster", palette="tab10", s=80)
plt.title("GMM ile Müşteri Segmentasyonu (PCA ile)")
plt.show()
