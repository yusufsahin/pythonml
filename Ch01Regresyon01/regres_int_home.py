import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 1. Veriyi yükle
df = pd.read_csv("data/house_data.csv")

# 2. Eksik değer var mı? (varsa doldur)
print("Eksik değerler:\n", df.isnull().sum())
df = df.fillna(df.mean(numeric_only=True))

# 3. Kategorik değişkenleri sayısala çevir (one-hot encoding)
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. X ve y ayır
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

# 5. Eğitim ve test ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Modeli kur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Tahmin ve başarı ölçümü
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Güncel ve uyumlu RMSE
print(f"\nR² skoru: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {rmse:,.0f}")

# 8. Modelin katsayıları
print("\nModel Katsayıları:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.1f}")

print(f"Sabit (intercept): {model.intercept_:.1f}")

# 9. Gerçek bir satırı formülle açıklama (ilk test örneği ile)
example_idx = X_test.index[0]
example_row = X_test.loc[example_idx]
gercek_fiyat = y_test.loc[example_idx]

print("\n--- Örnek Satır ile Tahminin Formülle Açıklanması ---")
print("Girdi değerleri:")
for col, val in example_row.items():
    print(f"{col}: {val}")

# Formül ile tahmini hesapla (manuel toplama)
tahmin_formul = model.intercept_
for col, val in example_row.items():
    tahmin_formul += val * model.coef_[list(X.columns).index(col)]
print(f"\nModelin formülle verdiği tahmin: {tahmin_formul:,.0f} TL")
print(f"Gerçek fiyat: {gercek_fiyat:,.0f} TL")

# 10. (İsteğe bağlı) Sonuç görselleştirme
plt.scatter(y_test, y_pred, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Lineer Regresyon: Gerçek vs Tahmin")
plt.tight_layout()
plt.show()