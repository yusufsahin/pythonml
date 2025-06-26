import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Veri seti: Yaş (X), Fiyat (y)
X = np.array([1,2,3,4,5,6,7,8,9,10,7,4]).reshape(-1,1)
y = np.array([800,720,690,610,590,500,480,450,410,370,900,200])  # 900 ve 200 outlier
# Klasik Lineer Regresyon
lr = LinearRegression()
lr.fit(X, y)

# RANSAC Regresyon
ransac = RANSACRegressor(LinearRegression(), residual_threshold=50)
ransac.fit(X, y)

# Belirli yaşlar için tahminler
test_ages = np.array([2, 5, 8, 10]).reshape(-1,1)
classic_preds = lr.predict(test_ages)
ransac_preds = ransac.predict(test_ages)

print("Yaş\tKlasik Regresyon\tRANSAC Regresyon")
for age, pred1, pred2 in zip(test_ages.ravel(), classic_preds, ransac_preds):
    print(f"{age}\t{pred1:.1f} bin TL\t\t{pred2:.1f} bin TL")

# Tüm yaşlar için çizgi (grafik için)
x_line = np.linspace(1, 10, 100).reshape(-1,1)
y_lr = lr.predict(x_line)
y_ransac = ransac.predict(x_line)

# Görselleştirme
plt.figure(figsize=(8,5))
plt.scatter(X, y, color="black", label="Gerçek Veri (outlier dahil)")
plt.plot(x_line, y_lr, color="blue", label="Klasik Regresyon")
plt.plot(x_line, y_ransac, color="red", label="RANSAC (Sağlam)")
plt.xlabel("Araba Yaşı (yıl)")
plt.ylabel("Fiyat (bin TL)")
plt.title("RANSAC ile Sağlam Regresyon (Gerçekçi Outlier Örneği)")
plt.legend()
plt.grid(True)
plt.show()
