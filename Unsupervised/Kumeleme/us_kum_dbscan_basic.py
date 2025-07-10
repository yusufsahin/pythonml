
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 1. VERİ YÜKLE
df = pd.read_csv("data/Mall_Customers.csv")

# 2. SADECE GELİR & HARCAMA SEÇ
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 3. ÖLÇEKLE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. DBSCAN UYGULA
dbscan = DBSCAN(eps=0.5, min_samples=5)
df["Cluster"] = dbscan.fit_predict(X_scaled)

# 5. GÖRSELLEŞTİR
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", palette="Set1", s=80)
plt.title("DBSCAN Basic Clustering")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Küme Sayıları
print("Küme Dağılımı:")
print(df["Cluster"].value_counts().sort_index())