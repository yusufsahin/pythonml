import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# 1. Veriyi oku
df = pd.read_csv("data/Ice_cream_selling_data.csv")
print("Veri seti ilk 5 satır:\n", df.head())
print("\nVeri seti sütun adları:", df.columns.tolist())

# 2. Bağımsız (X) ve bağımlı (y) değişkenleri seç
# 1. sütun: Bağımsız değişken (ör: sıcaklık), 2. sütun: Bağımlı değişken (ör: dondurma satışı)
X = df.iloc[:, 0].values.reshape(-1, 1)  # (n, 1) boyutunda olmalı
y = df.iloc[:, 1].values                 # (n,) boyutunda

# 3. Doğrusal (Linear) regresyon modeli kur ve eğit
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)  # Her X için lineer modelin tahmini

# 4. Polinomial özellikler üret (derece=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 5. Polinomial regresyon modeli kur ve eğit
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)  # Her X için polinomial modelin tahmini

# 6. Model başarı metrikleri (R2 ve RMSE)
print("\nLinear Regression R2:", r2_score(y, y_pred_linear))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y, y_pred_linear)))
print("\nPolynomial Regression (degree=2) R2:", r2_score(y, y_pred_poly))
print("Polynomial Regression RMSE:", np.sqrt(mean_squared_error(y, y_pred_poly)))

# 7. Polinomial modelin katsayıları (formülünü görmek için)
print("\nPolinomial Model Katsayıları:")
print("Sabit (b0):", poly_model.intercept_)
print("b1:", poly_model.coef_[1])
print("b2:", poly_model.coef_[2])

# 8. Tahminleri ve gerçek değerleri tablo olarak karşılaştır (ilk 10 satır)
df_tahmin = pd.DataFrame({
    df.columns[0]: X.flatten(),
    "Gerçek Satış": y,
    "Lineer Tahmin": y_pred_linear,
    "Polinomial Tahmin": y_pred_poly
})
print("\nİlk 10 satır için karşılaştırma:")
print(df_tahmin.head(10))

# 9. Konsolda ilk 5 gözlem için karşılaştırma
print("\nİlk 5 gözlem için tahminler:")
for i in range(5):
    print(f"{df.columns[0]}={X[i][0]} - Gerçek Satış: {y[i]}, Lineer Tahmin: {y_pred_linear[i]:.2f}, Polinomial Tahmin: {y_pred_poly[i]:.2f}")

# 10. Grafikle karşılaştır
plt.scatter(X, y, color='red', label='Gerçek Veri')
plt.plot(X, y_pred_linear, color='green', label='Lineer Regresyon')
plt.plot(X, y_pred_poly, color='blue', linestyle='dashed', label='Polinomial Regresyon (2. Derece)')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("Dondurma Satışı - Polinomial Regresyon Örneği")
plt.legend()
plt.tight_layout()
plt.show()
