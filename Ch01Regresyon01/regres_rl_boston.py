import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Tüm sütunları ve geniş tabloyu eksiksiz görebilmek için:
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 10)

# 1. Veri setini yükle
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
    "TAX", "PTRATIO", "B", "LSTAT"
]
df = pd.DataFrame(data, columns=columns)
df["MEDV"] = target

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# 2. Lineer, Ridge, Lasso modelleri tüm feature'larla eğit
models = {
    "Lineer": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000)
}
tahminler = {}
for isim, model in models.items():
    model.fit(X, y)
    tahminler[isim] = model.predict(X)

# 3. Polinomial Regresyon (örnek için sadece 'RM' feature'ı ile, 2. derece)
X_rm = df[["RM"]].values  # Ortalama oda sayısı
poly = PolynomialFeatures(degree=2)
X_rm_poly = poly.fit_transform(X_rm)
poly_model = LinearRegression().fit(X_rm_poly, y)
y_pred_poly = poly_model.predict(X_rm_poly)

# 4. İlk 10 gözlem için karşılaştırmalı tablo (tüm tahminler)
df_tahmin = pd.DataFrame({
    "Gerçek Fiyat (MEDV)": y.values[:10],
    "Lineer Tahmin": tahminler["Lineer"][:10],
    "Ridge Tahmin": tahminler["Ridge"][:10],
    "Lasso Tahmin": tahminler["Lasso"][:10],
    "Polinomial Tahmin (RM, d=2)": y_pred_poly[:10]
})
print("\nİlk 10 satır için tahmin karşılaştırması:\n")
print(df_tahmin.to_string(index=False))  # Tüm sütunları eksiksiz gör

# (İstersen satır satır da gösterebilirsin:)
print("\nDetaylı satır satır karşılaştırma:")
for i in range(10):
    print(f"Satır {i+1}: Gerçek={df_tahmin['Gerçek Fiyat (MEDV)'][i]:.2f}, "
          f"Lineer={df_tahmin['Lineer Tahmin'][i]:.2f}, "
          f"Ridge={df_tahmin['Ridge Tahmin'][i]:.2f}, "
          f"Lasso={df_tahmin['Lasso Tahmin'][i]:.2f}, "
          f"Polinomial={df_tahmin['Polinomial Tahmin (RM, d=2)'][i]:.2f}")

# 5. Grafikle tüm tahminleri ve gerçek değerleri karşılaştır (ilk 50 gözlem için)
df_tahmin_50 = pd.DataFrame({
    "Gerçek Fiyat (MEDV)": y.values[:50],
    "Lineer Tahmin": tahminler["Lineer"][:50],
    "Ridge Tahmin": tahminler["Ridge"][:50],
    "Lasso Tahmin": tahminler["Lasso"][:50],
    "Polinomial Tahmin (RM, d=2)": y_pred_poly[:50]
})



plt.figure(figsize=(15, 7))
plt.plot(df_tahmin_50.index, df_tahmin_50["Gerçek Fiyat (MEDV)"], "o-", label="Gerçek Değer", color="black")
plt.plot(df_tahmin_50.index, df_tahmin_50["Lineer Tahmin"], "g--", label="Lineer Regresyon")
plt.plot(df_tahmin_50.index, df_tahmin_50["Ridge Tahmin"], "b--", label="Ridge Regresyon")
plt.plot(df_tahmin_50.index, df_tahmin_50["Lasso Tahmin"], "r--", label="Lasso Regresyon")
plt.plot(df_tahmin_50.index, df_tahmin_50["Polinomial Tahmin (RM, d=2)"], "m--", label="Polinomial Regresyon (RM, d=2)")
plt.xlabel("Gözlem (index)")
plt.ylabel("Ev Fiyatı (MEDV, 1000$)")
plt.title("Lineer, Ridge, Lasso, Polinomial Regresyon Tahmin Karşılaştırması (Boston Housing, İlk 50 Gözlem)")
plt.legend()
plt.tight_layout()
plt.show()



# 6. (Ekstra) Model başarı skorları (R2)
print("\nR2 skorları:")
print("Lineer:    ", r2_score(y, tahminler["Lineer"]))
print("Ridge:     ", r2_score(y, tahminler["Ridge"]))
print("Lasso:     ", r2_score(y, tahminler["Lasso"]))
print("Polinomial (sadece RM, d=2):", r2_score(y, y_pred_poly))


print("\n--- Model Katsayıları ve Formülleri ---\n")

def linear_formula(model, columns, name="Model"):
    intercept = model.intercept_
    coefs = model.coef_
    terms = [f"{coef:.3f}*{col}" for coef, col in zip(coefs, columns)]
    formula = f"{name}: y = {intercept:.3f} + " + " + ".join(terms)
    return formula

# Lineer
print("Lineer Regresyon Katsayıları:")
for name, coef in zip(X.columns, models["Lineer"].coef_):
    print(f"{name:>8}: {coef:.3f}")
print("Sabit terim (intercept): {:.3f}".format(models["Lineer"].intercept_))
print("Formül:")
print(linear_formula(models["Lineer"], X.columns, "Lineer"))

print("\nRidge Regresyon Katsayıları:")
for name, coef in zip(X.columns, models["Ridge"].coef_):
    print(f"{name:>8}: {coef:.3f}")
print("Sabit terim (intercept): {:.3f}".format(models["Ridge"].intercept_))
print("Formül:")
print(linear_formula(models["Ridge"], X.columns, "Ridge"))

print("\nLasso Regresyon Katsayıları:")
for name, coef in zip(X.columns, models["Lasso"].coef_):
    print(f"{name:>8}: {coef:.3f}")
print("Sabit terim (intercept): {:.3f}".format(models["Lasso"].intercept_))
print("Formül:")
print(linear_formula(models["Lasso"], X.columns, "Lasso"))

# Polinomial (sadece RM, d=2)
print("\nPolinomial Regresyon (RM, d=2) Katsayıları:")
coef = poly_model.coef_
intercept = poly_model.intercept_
print(f"Sabit terim (intercept): {intercept:.3f}")
print(f"RM Katsayısı: {coef[1]:.3f}")
print(f"RM^2 Katsayısı: {coef[2]:.3f}")
print("Formül:")
print(f"Polinomial: y = {intercept:.3f} + {coef[1]:.3f}*RM + {coef[2]:.3f}*RM^2")

