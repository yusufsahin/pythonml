import numpy as np  # Sayısal işlemler için
import pandas as pd  # Tablo şeklinde veri işlemleri için
import matplotlib.pyplot as plt  # Grafik çizmek için

# scikit-learn'den gerekli modüller
from sklearn.linear_model import ElasticNet  # Model tipi
from sklearn.datasets import fetch_california_housing  # Hazır veri seti
from sklearn.model_selection import train_test_split, GridSearchCV  # Veri bölme, parametre optimizasyonu
from sklearn.preprocessing import StandardScaler  # Özellikleri aynı ölçeğe getirme
from sklearn.metrics import r2_score, mean_squared_error  # Model başarısı ölçümü

# 1. Veri Setini Yükle ve Keşfet
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

print("İlk 5 satır:\n", X.head())
print("\nHedef değişkenin (fiyat) özet istatistikleri:\n", y.describe())
print("\nEksik değer var mı?\n", X.isnull().sum())

# 2. Özellikleri Ölçekle (Standardizasyon)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Eğitim/Test Ayırımı (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("\nEğitim veri seti boyutu:", X_train.shape)
print("Test veri seti boyutu:", X_test.shape)

# 4. ElasticNet için Parametre Aralığı Tanımla
param_grid = {
    "alpha": np.logspace(-2, 1, 10),     # 0.01 ile 10 arası
    "l1_ratio": np.linspace(0.1, 0.9, 9) # 0.1'den 0.9'a arası
}
model = ElasticNet(max_iter=10000, random_state=42)

# 5. GridSearchCV ile Otomatik En İyi Parametre Seçimi
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                  # 5 katlı çapraz doğrulama
    scoring='r2',          # R2 başarısı
    n_jobs=-1              # Tüm işlemcilerle hızlı deneme
)
grid.fit(X_train, y_train)

print("\nEn iyi alpha:", grid.best_params_['alpha'])
print("En iyi l1_ratio:", grid.best_params_['l1_ratio'])
print("En iyi cross-validation R2 skoru:", grid.best_score_)

# 6. Test Setinde En İyi Modelin Performansı
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nTest setinde R2 skoru:", r2)
print("Test setinde RMSE:", rmse)

# 7. Katsayı Analizi
feature_importance = pd.Series(
    best_model.coef_, index=X.columns
).sort_values(key=lambda x: np.abs(x), ascending=False)
print("\nÖzellik önemleri (büyüklüğe göre):\n", feature_importance)
print("\nSıfırlanan feature'lar:", list(feature_importance[feature_importance == 0].index))

# 8. İlk 10 Tahmin ve Gerçek Karşılaştırması
compare_df = pd.DataFrame({
    "Gerçek Değer": y_test.values[:10],
    "Tahmin": np.round(y_pred[:10], 2)
})
print("\nİlk 10 tahmin ve gerçek değer:\n", compare_df)

# Orijinal (ham) feature'lar ile kullanılacak formülün katsayılarını ve sabit terimini hesapla
coefs_original = best_model.coef_ / scaler.scale_
intercept_original = (
    best_model.intercept_
    - np.sum(scaler.mean_ * best_model.coef_ / scaler.scale_)
)

print("\nModelin orijinal (ham) değişkenlerle matematiksel formülü:")
print(f"y = {intercept_original:.4f} + ", end="")
print(" + ".join([
    f"({coef:.4f} * {name})"
    for coef, name in zip(coefs_original, X.columns)
]))



print("Model formülü:")
print(f"y = {best_model.intercept_:.4f} + ", end="")
print(" + ".join([
    f"({coef:.4f} * {name})"
    for coef, name in zip(best_model.coef_, X.columns)
]))

# 9. Tahmin-Grafik
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Gerçek Değer")
plt.ylabel("Tahmin")
plt.title("ElasticNet: Gerçek vs Tahmin (Test Seti)")
plt.tight_layout()
plt.show()
