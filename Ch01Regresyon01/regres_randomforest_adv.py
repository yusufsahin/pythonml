# 1. Kütüphaneleri yükle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2. Rastgelelik kontrolü – Deneyin tekrarlanabilir olması için
RANDOM_STATE = 42
# 3. Veri setini yükle ve keşfet
data = fetch_california_housing()
X = data.data[:5000]
y = data.target[:5000]
print("Özellik isimleri:", data.feature_names)
print("Veri boyutu:", X.shape)
# 4. Eğitim/test ayırımı (best practice: önce ayır, sonra preprocessing yapılır!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
# 5. (Opsiyonel) Eksik veri ve aykırı değer analizi – Gelişmiş projelerde mutlaka yapılmalı
# Bu veri setinde eksik yok, gerçek projede X_train.isnull().sum() vs. bakılır.

# 6. Hiperparametre optimizasyonu için GridSearchCV ayarla
# Her parametre modelin başarısını büyük oranda etkiler!
param_grid = {
    'n_estimators': [100, 200],       # Ormandaki ağaç sayısı
    'max_depth': [None, 10, 20],      # Maksimum ağaç derinliği
    'min_samples_split': [2, 5, 10],  # Dallanma için gereken min. örnek
    'min_samples_leaf': [1, 2, 4],    # Yaprakta olması gereken min. örnek
    'max_features': ['sqrt', 'log2']  # Split için rastgele özellik seçimi
}
# cv=3: 3 katlı çapraz doğrulama, n_jobs=-1: Tüm CPU çekirdeklerini kullan
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # MSE'yi minimize edecek şekilde
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)   # Sadece eğitim verisiyle fit edilir
print("En iyi parametreler:", grid_search.best_params_)
# 7. En iyi modeli al ve tüm eğitim verisiyle yeniden eğit
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# 8. Test setinde tahmin yap ve başarıyı ölç
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Test MSE:", mse)
print("Test R2:", r2)

# 9. Çapraz doğrulama ile modelin genel başarısını ölç
# 5 katlı çapraz doğrulama, skoru pozitif için - ile çarpılır
cv_scores = -cross_val_score(
    best_rf, X_train, y_train,
    scoring='neg_mean_squared_error', cv=5
)
print("Çapraz doğrulama ortalama MSE:", np.mean(cv_scores))

# 10. Özellik önemlerini görselleştir (modelin hangi değişkenlere nasıl karar verdiği)
importances = best_rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
plt.title("Özellik Önemleri")
plt.bar(range(X.shape[1]), importances[sorted_idx], align="center")
plt.xticks(range(X.shape[1]), np.array(data.feature_names)[sorted_idx], rotation=45)
plt.tight_layout()
plt.show()

# 11. Gerçek ve tahmin değerlerini scatter plot ile karşılaştır
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.3, color="royalblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Gerçek Ev Fiyatı ($100,000)")
plt.ylabel("Tahmin Edilen Ev Fiyatı ($100,000)")
plt.title("Random Forest: Gerçek vs Tahmin")
plt.grid(True)
plt.show()

# 12. Tahmin hatası histogramı (hata dağılımı analizi)
errors = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.hist(errors, bins=30, color="orange", edgecolor="black")
plt.xlabel("Tahmin Hatası")
plt.ylabel("Frekans")
plt.title("Random Forest Tahmin Hata Dağılımı")
plt.grid(True)
plt.show()
