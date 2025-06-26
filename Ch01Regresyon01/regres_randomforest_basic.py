# 1. Gerekli kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sklearn
print("Kullanılan scikit-learn sürümü:", sklearn.__version__)

# 2. Veri setini yükle (Ames Housing)
data = fetch_openml(name="house_prices", as_frame=True)
df = data.frame.copy()

# 3. Giriş (X) ve hedef (y) değişkenlerini ayır
y = df["SalePrice"].astype(float)
X = df.drop("SalePrice", axis=1)

print("🔍 Veri şekli:", X.shape)

# 4. Sayısal ve kategorik sütunları belirle
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 5. Eksik veri analizi
print("\n🧼 Eksik veri oranı en yüksek ilk 10 sütun:")
missing = X.isnull().mean().sort_values(ascending=False)
print(missing.head(10))

# 6. Sayısal ve kategorik veri işleme pipeline'ları
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # ✅ sklearn 1.6.1 uyumlu
])

# 7. Tüm veri ön işleme adımlarını birleştir
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 8. Eğitim/Test ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Model pipeline (ön işleme + random forest)
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))
])

# 10. Modeli eğit
model.fit(X_train, y_train)

# 11. Tahmin yap
y_pred = model.predict(X_test)

# 12. Performans metrikleri
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performansı:")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# 13. Tahmin vs Gerçek görselleştirme
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("🎯 Gerçek vs Tahmin Edilen Konut Fiyatı")
plt.tight_layout()
plt.show()

# 14. Modeli kaydet
joblib.dump(model, "rf_ames_model.pkl")
print("\n✅ Model başarıyla 'rf_ames_model.pkl' olarak kaydedildi.")
