import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Veri yükle
df = pd.read_csv("data/Advertising.csv", index_col=0)

# 2. Eksik değer kontrolü ve doldurma (örnek amaçlı, veri setinde eksik yok)
print("Eksik değerler:\n", df.isnull().sum())
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median() if df[col].dtype != 'O' else 'unknown')

# 3. Aykırı değer temizliği (ör: TV, Radio, Newspaper harcamalarında uç değerleri temizle)
for col in ["TV", "Radio", "Newspaper"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 4. Feature Engineering: Harcama başına satış ve toplam harcama gibi yeni sütunlar ekle
df["sales_per_tv"] = df["Sales"] / (df["TV"] + 1)
df["sales_per_radio"] = df["Sales"] / (df["Radio"] + 1)
df["total_spend"] = df["TV"] + df["Radio"] + df["Newspaper"]

# 5. Özellik ve hedef değişkenleri ayır
X = df.drop("Sales", axis=1)
y = df["Sales"]

# 6. Özellik ölçekleme (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 7. Eğitim/test ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Modeli kur & eğit
model = LinearRegression()
model.fit(X_train, y_train)

# 9. Tahmin ve metrikler
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nR2 skoru: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: {rmse:.2f}")

# 10. Model katsayıları
print("\nModel Katsayıları:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.3f}")
print(f"Sabit (intercept): {model.intercept_:.3f}")

# 11. İlk test örneğinde adım adım tahmin
example_row = X_test.iloc[0]
gercek_satis = y_test.iloc[0]

print("\n--- Örnek Satır Tahmin Açıklaması ---")
print("Girdi değerleri (ölçeklenmiş):")
for col, val in example_row.items():
    print(f"{col}: {val:.3f}")

tahmin_formul = model.intercept_
for col, val in example_row.items():
    tahmin_formul += val * model.coef_[list(X.columns).index(col)]
print(f"\nModelin formülle verdiği tahmin: {tahmin_formul:.2f}")
print(f"Gerçek satış: {gercek_satis:.2f}")

# 12. Görselleştirme
plt.scatter(y_test, y_pred, color='green', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Gerçek Satış")
plt.ylabel("Tahmin Edilen Satış")
plt.title("Advanced Lineer Regresyon: Gerçek vs Tahmin")
plt.tight_layout()
plt.show()
