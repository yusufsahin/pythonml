# Gerekli kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Adım 1: Veri seti oluştur (3 kümeli yapay veri)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

# Adım 2: GMM modelini oluştur
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Adım 3: Küme tahminleri yap
labels = gmm.predict(X)

# Adım 4: Her noktanın kümelere ait olma olasılığı (soft clustering)
probs = gmm.predict_proba(X)

# Adım 5: Sonucu görselleştir
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.title("Gaussian Mixture Model (GMM) Kümeleme")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.grid(True)
plt.show()
