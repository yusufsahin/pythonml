import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# 1. VERİYİ YÜKLE ve HAZIRLA
RANDOM_STATE = 42
data = fetch_california_housing()
X = data.data[:2000]
y = data.target[:2000]
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# 2. ELASTICNET (GridSearch ile)
enet_param_grid = {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.1, 0.5, 0.9]}
enet_search = GridSearchCV(
    ElasticNet(max_iter=10000),
    param_grid=enet_param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=0
)
enet_search.fit(X_train_scaled, y_train_scaled)
best_enet = enet_search.best_estimator_
y_pred_enet_scaled = best_enet.predict(X_test_scaled)
y_pred_enet = scaler_y.inverse_transform(y_pred_enet_scaled.reshape(-1, 1)).ravel()
mse_enet = mean_squared_error(y_test, y_pred_enet)
r2_enet = r2_score(y_test, y_pred_enet)
coef_enet = best_enet.coef_
intercept_enet = best_enet.intercept_

# 3. SVR (GridSearch ile)
svr_param_grid = {
    "kernel": ["rbf"], "C": [1, 10], "epsilon": [0.1], "gamma": ["scale"]
}
svr_search = GridSearchCV(
    SVR(),
    param_grid=svr_param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=0
)
svr_search.fit(X_train_scaled, y_train_scaled)
best_svr = svr_search.best_estimator_
y_pred_svr_scaled = best_svr.predict(X_test_scaled)
y_pred_svr = scaler_y.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).ravel()
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# 4. KATSAYI & FORMÜLLERİN YAZDIRILMASI
elasticnet_formul = "ElasticNet (ölçekli):\ny_hat = {:.4f}".format(intercept_enet)
for fname, coef in zip(feature_names, coef_enet):
    elasticnet_formul += " + ({:.4f} * {})".format(coef, fname)
print(elasticnet_formul)
print("\nSVR (RBF kernel) formülü:")
print("y_hat = sum_i [alpha_i * K(x_i, x)] + b")
print("Support vector sayısı:", best_svr.support_vectors_.shape[0])
print("Bias (intercept_):", best_svr.intercept_[0])

# 5. SONUÇLARI YAZDIR
print("\nSkorlar (Test seti):")
print("ElasticNet  - Test MSE: {:.4f}, R2: {:.4f}".format(mse_enet, r2_enet))
print("SVR (RBF)   - Test MSE: {:.4f}, R2: {:.4f}".format(mse_svr, r2_svr))

# 6. GÖRSELLEŞTİRME (plt ile)
plt.figure(figsize=(15, 5))
# Gerçek vs Tahmin (ElasticNet)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_enet, alpha=0.5, color="crimson", label="ElasticNet")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Doğru Doğruya")
plt.xlabel("Gerçek Ev Fiyatı ($100,000)")
plt.ylabel("Tahmin (ElasticNet)")
plt.title("ElasticNet: Gerçek vs Tahmin")
plt.legend()
plt.grid(True)
# Gerçek vs Tahmin (SVR)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_svr, alpha=0.5, color="royalblue", label="SVR (RBF)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Doğru Doğruya")
plt.xlabel("Gerçek Ev Fiyatı ($100,000)")
plt.ylabel("Tahmin (SVR)")
plt.title("SVR: Gerçek vs Tahmin")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Hata Dağılımı
plt.figure(figsize=(15, 4))
errors_enet = y_test - y_pred_enet
errors_svr  = y_test - y_pred_svr
plt.subplot(1, 2, 1)
plt.hist(errors_enet, bins=25, color="crimson", edgecolor="black", alpha=0.7)
plt.title("ElasticNet Tahmin Hata Dağılımı")
plt.xlabel("Tahmin Hatası")
plt.ylabel("Frekans")
plt.grid(True)
plt.subplot(1, 2, 2)
plt.hist(errors_svr, bins=25, color="royalblue", edgecolor="black", alpha=0.7)
plt.title("SVR Tahmin Hata Dağılımı")
plt.xlabel("Tahmin Hatası")
plt.ylabel("Frekans")
plt.grid(True)
plt.tight_layout()
plt.show()

# İlk 20 örnek karşılaştırma
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(range(20), y_test[:20], label="Gerçek", marker='o', color='black')
plt.plot(range(20), y_pred_enet[:20], label="ElasticNet Tahmin", marker='x', color='crimson')
plt.title("ElasticNet - İlk 20 Ev: Gerçek vs Tahmin")
plt.xlabel("Ev No")
plt.ylabel("Fiyat ($100,000)")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(20), y_test[:20], label="Gerçek", marker='o', color='black')
plt.plot(range(20), y_pred_svr[:20], label="SVR Tahmin", marker='x', color='royalblue')
plt.title("SVR - İlk 20 Ev: Gerçek vs Tahmin")
plt.xlabel("Ev No")
plt.ylabel("Fiyat ($100,000)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
