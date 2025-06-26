import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Veriyi yükle
df = pd.read_csv("data/house_data.csv")

# 2. Eksik veri analizi ve doldurma
print("Eksik değerler:\n", df.isnull().sum())

# Sayısal ve kategorik kolonları ayır
sayi_kolonlar = df.select_dtypes(include=[np.number]).columns.tolist()
kat_kolonlar = df.select_dtypes(include=['object']).columns.tolist()

# Sayısal eksikleri ortanca ile doldur
for col in sayi_kolonlar:
    df[col] = df[col].fillna(df[col].median())

# Kategorik eksikleri 'unknown' ile doldur
for col in kat_kolonlar:
    df[col] = df[col].fillna('unknown')

# 3. Aykırı değer temizliği (örnek: 'area' için IQR yöntemi)
Q1 = df['area'].quantile(0.25)
Q3 = df['area'].quantile(0.75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5 * IQR
ust_sinir = Q3 + 1.5 * IQR
df = df[(df['area'] >= alt_sinir) & (df['area'] <= ust_sinir)]

# 4. Kategorik değişkenleri one-hot encoding ile sayısala çevir
df_encoded = pd.get_dummies(df, drop_first=True)

# 5. Feature Engineering: m2 başına fiyat gibi yeni bir sütun ekle
df_encoded['price_per_m2'] = df_encoded['price'] / df_encoded['area']

# 6. Özellik ve hedef değişken ayır
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

# 7. Özellik ölçekleme (StandardScaler ile)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 8. Eğitim ve test setine ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Modeli kur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 10. Tahmin ve başarı ölçümü
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nR² skoru: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {rmse:,.0f}")

# 11. Model katsayıları
print("\nModel Katsayıları:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.2f}")
print(f"Sabit (intercept): {model.intercept_:.2f}")

# 12. Gerçek bir satırı formülle açıklama (ilk test örneğiyle)
example_idx = X_test.index[0]
example_row = X_test.loc[example_idx]
gercek_fiyat = y_test.loc[example_idx]

print("\n--- Örnek Satır ile Tahminin Formülle Açıklanması ---")
print("Girdi değerleri (ölçeklenmiş):")
for col, val in example_row.items():
    print(f"{col}: {val:.3f}")

tahmin_formul = model.intercept_
for col, val in example_row.items():
    tahmin_formul += val * model.coef_[list(X.columns).index(col)]
print(f"\nModelin formülle verdiği tahmin: {tahmin_formul:,.0f} TL")
print(f"Gerçek fiyat: {gercek_fiyat:,.0f} TL")

# 13. (Opsiyonel) Sonuçları görselleştir
plt.scatter(y_test, y_pred, color="blue", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Lineer Regresyon: Gerçek vs Tahmin")
plt.tight_layout()
plt.show()
