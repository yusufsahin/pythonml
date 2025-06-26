from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 0. Temel veri keşfi: Kaç satır/feature ve ilk 5 örnek
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

print(f"Toplam {X.shape[0]} örnek (satır), {X.shape[1]} feature (sütun) var.\n")

df = pd.DataFrame(X, columns=feature_names)
print("İlk 5 satır (özellikler):")
print(df.head())
print("\nİlk 5 hedef (y):")
print(y[:5])
print("="*60)

# 2. Modelleri tanımla ve eğit
models = {
    "Lineer": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000)
}
tahminler = {}
for isim, model in models.items():
    model.fit(X, y)
    tahminler[isim] = model.predict(X)

# 3. Konsolda ilk 10 gözlem için karşılaştırmalı tablo
df_tahmin = pd.DataFrame({
    "Gerçek": y[:10],
    "Lineer Tahmin": tahminler["Lineer"][:10],
    "Ridge Tahmin": tahminler["Ridge"][:10],
    "Lasso Tahmin": tahminler["Lasso"][:10]
})
print("\nİlk 10 gözlem için tahmin karşılaştırması:\n")
print(df_tahmin.round(2).to_string(index=False))

# 4. Detaylı satır satır göster
print("\nDetaylı karşılaştırma (ilk 10):")
for i in range(10):
    print(f"Satır {i+1}: Gerçek={df_tahmin['Gerçek'][i]:.2f}, "
          f"Lineer={df_tahmin['Lineer Tahmin'][i]:.2f}, "
          f"Ridge={df_tahmin['Ridge Tahmin'][i]:.2f}, "
          f"Lasso={df_tahmin['Lasso Tahmin'][i]:.2f}")

# 5. Grafik: İlk 50 gözlem için gerçek ve tahminler
df_tahmin_50 = pd.DataFrame({
    "Gerçek": y[:50],
    "Lineer Tahmin": tahminler["Lineer"][:50],
    "Ridge Tahmin": tahminler["Ridge"][:50],
    "Lasso Tahmin": tahminler["Lasso"][:50]
})

plt.figure(figsize=(15, 7))
plt.plot(df_tahmin_50.index, df_tahmin_50["Gerçek"], "o-", label="Gerçek Değer", color="black")
plt.plot(df_tahmin_50.index, df_tahmin_50["Lineer Tahmin"], "g--", label="Lineer Regresyon")
plt.plot(df_tahmin_50.index, df_tahmin_50["Ridge Tahmin"], "b--", label="Ridge Regresyon")
plt.plot(df_tahmin_50.index, df_tahmin_50["Lasso Tahmin"], "r--", label="Lasso Regresyon")
plt.xlabel("Gözlem (index)")
plt.ylabel("Diyabet Skoru")
plt.title("Lineer, Ridge, Lasso Regresyon Tahmin Karşılaştırması (Diabetes, İlk 50 Gözlem)")
plt.legend()
plt.tight_layout()
plt.show()


# 6. Katsayıları göster
print("\nModel katsayıları (özellikler sırasıyla):")
df_coef = pd.DataFrame(
    {isim: model.coef_ for isim, model in models.items()},
    index=feature_names
)
print(df_coef.round(3))

# 7. R2 skorları
print("\nR2 skorları:")
print("Lineer: ", r2_score(y, tahminler["Lineer"]))
print("Ridge:  ", r2_score(y, tahminler["Ridge"]))
print("Lasso:  ", r2_score(y, tahminler["Lasso"]))
