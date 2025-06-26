import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Küçük, gerçekçi veri seti
# Sıcaklık (°C) ve Elektrik Tüketimi (kWh)
X = np.array([-5, 0, 10, 20, 30, 40]).reshape(-1, 1)
y = np.array([120, 80, 40, 20, 45, 80])

# 2. Doğrusal regresyon
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# 3. Polinomial regresyon (derece=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# 4. Model katsayılarını yazdır
print("Lineer Regresyon: y = {:.2f} + {:.2f} * X".format(lin_reg.intercept_, lin_reg.coef_[0]))
print("Polinomial Regresyon: y = {:.2f} + {:.2f} * X + {:.2f} * X^2".format(
    poly_reg.intercept_, poly_reg.coef_[1], poly_reg.coef_[2]
))

# 5. Tahminleri tablo olarak göster
df_tahmin = pd.DataFrame({
    "Sıcaklık (°C)": X.flatten(),
    "Gerçek Elektrik": y,
    "Lineer Tahmin": np.round(y_pred_lin, 2),
    "Polinomial Tahmin": np.round(y_pred_poly, 2)
})
print("\nTahmin Karşılaştırma Tablosu:\n", df_tahmin)

# 6. Grafikle görselleştir
plt.scatter(X, y, color='red', label="Gerçek Veri")
plt.plot(X, y_pred_lin, color='green', label="Lineer Regresyon")
plt.plot(X, y_pred_poly, color='blue', linestyle='--', label="Polinomial Regresyon (2. derece)")
plt.xlabel("Sıcaklık (°C)")
plt.ylabel("Elektrik Tüketimi (kWh)")
plt.title("Sıcaklık ve Elektrik Tüketimi - Polinomial Regresyon")
plt.legend()
plt.tight_layout()
plt.show()