from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
# Rastgelelik için seed (tekrarlanabilirlik)
RANDOM_STATE = 42

#1. Veri Setini Yükle
data = fetch_california_housing()
X = data.data[:2000]   # İlk 2000 örnek alınarak işlem hızlandırılır
y = data.target[:2000]
#2. Eğitim/Test Ayırımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

#3.Özellik Ölçekleme/ hedefi standartlaştırma - SVR için gerekli
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()


#4.Parametre Aralığı Tanımla / Grid (test amaçlı)
param_grid = {
    "kernel": ["rbf"],
    "C": [1, 10],
    "epsilon": [0.1],
    "gamma": ["scale"]
}

#5.GridSearchCV ile Otomatik En İyi Parametre Seçimi /Hiperparametre optimizasyonu
grid_search = GridSearchCV(
    estimator=SVR(),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_scaled, y_train_scaled)
#6.En iyi parametreleri ve modeli al
print("En iyi parametreler:", grid_search.best_params_)
best_svr = grid_search.best_estimator_
#7. Modeli tekrar eğit
best_svr.fit(X_train_scaled, y_train_scaled)
#8. Test seti üzerinde tahmin yap
y_pred_scaled = best_svr.predict(X_test_scaled)
#9.Tahminleri orijinal ölçeğe çevir
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

#10. Performans metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Test Mean Squared Error (MSE):", mse)
print("Test R² Score:", r2)

#11.GÖRSELLEŞTİRME 1: Gerçek vs Tahmin scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color="royalblue", label="Tahmin")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Doğru Doğruya")
plt.xlabel("Gerçek Ev Fiyatı ($100,000)")
plt.ylabel("Tahmin Edilen Ev Fiyatı ($100,000)")
plt.title("SVR: Gerçek vs Tahmin Değerleri")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#12.GÖRSELLEŞTİRME 2: Tahmin hatası dağılımı
errors = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.hist(errors, bins=25, color="orange", edgecolor="black")
plt.xlabel("Tahmin Hatası")
plt.ylabel("Frekans")
plt.title("SVR Tahmin Hata Dağılımı")
plt.grid(True)
plt.tight_layout()
plt.show()
#13.GÖRSELLEŞTİRME 3: İlk 20 örnek için karşılaştırma
plt.figure(figsize=(12, 5))
plt.plot(range(20), y_test[:20], label="Gerçek", marker='o')
plt.plot(range(20), y_pred[:20], label="Tahmin", marker='x')
plt.title("SVR - İlk 20 Ev: Gerçek vs Tahmin")
plt.xlabel("Ev No")
plt.ylabel("Fiyat ($100,000)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()