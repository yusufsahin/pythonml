import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 1. Veri Oluştur
X, y_true = make_blobs(n_samples=30, centers=3, cluster_std=1.0, random_state=42)

# Ham veri görselleştirme
plt.scatter(X[:, 0], X[:, 1])
plt.title("Ham Veri (30 Nokta, 3 Küme)")
plt.show()

# 2. Linkage matrisi oluştur (Ward yöntemi)
Z = linkage(X, method='ward')

# 3. Dendrogram çizimi
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrogram (Ward Linkage)")
plt.xlabel("Veri Noktası İndeksleri")
plt.ylabel("Mesafe")
plt.show()

# 4. Kümeleme sonucu: threshold=10'a göre kümeleri çıkar
labels = fcluster(Z, t=10, criterion='distance')

# 5. Sonuçları çiz
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Set1')
plt.title("Hierarchical Clustering Sonucu (3 Küme)")
plt.show()
