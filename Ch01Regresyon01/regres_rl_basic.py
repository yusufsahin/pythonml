import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
import matplotlib.pyplot as plt
# Küçük veri seti :[yaş,tecrübe,sertifika(0/1),maaş] ->maaş tahmini

X=np.array( [[25, 1, 0],
    [35, 10, 1],
    [45, 20, 0],
    [30, 3, 1],
    [50, 25, 0]])
y= np.array([3500, 7000, 9000, 4500, 11000]) # Maaşlar

#1. Doğrusal regresyon modeli oluşturma

lr= LinearRegression().fit(X,y)
y_pred_lr = lr.predict(X)

#2. Ridde Regresyon
ridge = Ridge(alpha=1.0).fit(X,y)
y_pred_ridge = ridge.predict(X)

#3.Lasso Regresyon
lasso=Lasso(alpha=1000).fit(X,y)
y_pred_lasso = lasso.predict(X)

# 4. Tahminlerin Tablo Olarak Gösterilmesi
df = pd.DataFrame({
    "Gerçek Maaş": y,
    "Lineer Tahmin": np.round(y_pred_lr, 2),
    "Ridge Tahmin": np.round(y_pred_ridge, 2),
    "Lasso Tahmin": np.round(y_pred_lasso, 2)
})
print("\nTahmin Karşılaştırması:\n")
print(df.to_string(index=False))
# 5. Grafikle Karşılaştırma
plt.figure(figsize=(10, 6))
plt.plot(y, 'ko-', label="Gerçek Maaş", linewidth=2)
plt.plot(y_pred_lr, 'g--o', label="Lineer Regresyon")
plt.plot(y_pred_ridge, 'b--s', label="Ridge Regresyon")
plt.plot(y_pred_lasso, 'r--^', label="Lasso Regresyon")
plt.xlabel("Gözlem (index)")
plt.ylabel("Maaş")
plt.title("Ridge & Lasso - Tahmin ve Gerçek Değer Karşılaştırması")
plt.legend()
plt.tight_layout()
plt.show()
