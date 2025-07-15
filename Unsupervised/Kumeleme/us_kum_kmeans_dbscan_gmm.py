import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

df = pd.read_csv("data/Mall_Customers.csv")
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=42)
dbscan = DBSCAN(eps=0.7, min_samples=5)
gmm = GaussianMixture(n_components=4, random_state=42)

kmeans_labels = kmeans.fit_predict(X_scaled)
dbscan_labels = dbscan.fit_predict(X_scaled)
gmm_labels = gmm.fit_predict(X_scaled)

df_vis = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df_vis["KMeans"] = kmeans_labels
df_vis["DBSCAN"] = dbscan_labels
df_vis["GMM"] = gmm_labels

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(data=df_vis, x="PCA1", y="PCA2", hue="KMeans", ax=axes[0], palette="tab10")
axes[0].set_title("K-Means")

sns.scatterplot(data=df_vis, x="PCA1", y="PCA2", hue="DBSCAN", ax=axes[1], palette="tab10")
axes[1].set_title("DBSCAN")

sns.scatterplot(data=df_vis, x="PCA1", y="PCA2", hue="GMM", ax=axes[2], palette="tab10")
axes[2].set_title("Gaussian Mixture Model")

plt.suptitle("Mall Customer Segmentation - KMeans, DBSCAN, GMM")
plt.tight_layout()
plt.show()
