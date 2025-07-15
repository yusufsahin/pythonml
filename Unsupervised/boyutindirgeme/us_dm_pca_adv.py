import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Yüksek boyutlu veri seti: Wine
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# 2. Normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA uygulaması
pca = PCA(n_components=0.95)  # %95 varyansı koruyacak kadar bileşen seç
X_pca = pca.fit_transform(X_scaled)

print(f"PCA ile kalan bileşen sayısı: {X_pca.shape[1]}")
print("Varyans oranları:", pca.explained_variance_ratio_)

# 4. PCA sonrası modelleme
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Doğruluk (PCA sonrası):", accuracy_score(y_test, y_pred))

# 5. KMeans ile Segmentasyon (Görselleştirme için sadece 2 PCA bileşeni al)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_2d)

# 6. Görselleştirme
df_vis = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
df_vis["Cluster"] = clusters
df_vis["Target"] = y

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_vis, x="PC1", y="PC2", hue="Cluster", style="Target", palette="tab10", s=80)
plt.title("PCA + KMeans ile Segmentasyon")
plt.grid(True)
plt.show()
