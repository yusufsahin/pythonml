import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer

# Veri setini yükle
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------- 1. Gradient Boosting Regressor (GBDT) ---------------------
from sklearn.ensemble import GradientBoostingRegressor

print("\n--- GradientBoostingRegressor ---")
# Hiperparametre tuning: n_estimators (ağaç sayısı), learning_rate (öğrenme oranı), max_depth (ağaç derinliği)
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5]
}
grid_gbdt = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid_gbdt.fit(X_train, y_train)
best_gbdt = grid_gbdt.best_estimator_
y_pred_gbdt = best_gbdt.predict(X_test)
print("Best Params:", grid_gbdt.best_params_)
print("Test MSE:", mean_squared_error(y_test, y_pred_gbdt))
print("Test R2 :", r2_score(y_test, y_pred_gbdt))

# Özellik önemi grafiği
plt.figure(figsize=(8,3))
plt.barh(np.array(fetch_california_housing().feature_names), best_gbdt.feature_importances_)
plt.title("GBDT Feature Importances")
plt.show()

# --------------------- 2. XGBoost Regressor ---------------------
from xgboost import XGBRegressor

print("\n--- XGBoostRegressor ---")
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}
grid_xgb = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
print("Best Params:", grid_xgb.best_params_)
print("Test MSE:", mean_squared_error(y_test, y_pred_xgb))
print("Test R2 :", r2_score(y_test, y_pred_xgb))

plt.figure(figsize=(8,3))
plt.barh(np.array(fetch_california_housing().feature_names), best_xgb.feature_importances_)
plt.title("XGBoost Feature Importances")
plt.show()

# --------------------- 3. LightGBM Regressor ---------------------
from lightgbm import LGBMRegressor

print("\n--- LightGBMRegressor ---")
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "num_leaves": [31, 50]
}
grid_lgbm = GridSearchCV(LGBMRegressor(random_state=42), param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid_lgbm.fit(X_train, y_train)
best_lgbm = grid_lgbm.best_estimator_
y_pred_lgbm = best_lgbm.predict(X_test)
print("Best Params:", grid_lgbm.best_params_)
print("Test MSE:", mean_squared_error(y_test, y_pred_lgbm))
print("Test R2 :", r2_score(y_test, y_pred_lgbm))

plt.figure(figsize=(8,3))
plt.barh(np.array(fetch_california_housing().feature_names), best_lgbm.feature_importances_)
plt.title("LightGBM Feature Importances")
plt.show()

# --------------------- 4. CatBoost Regressor ---------------------
from catboost import CatBoostRegressor

print("\n--- CatBoostRegressor ---")
param_grid = {
    "iterations": [100, 200],
    "learning_rate": [0.05, 0.1],
    "depth": [4, 6]
}
# CatBoost'un GridSearchCV ile kullanımı RandomizedSearchCV ile daha sağlıklı olur.
from sklearn.model_selection import RandomizedSearchCV
grid_cat = RandomizedSearchCV(CatBoostRegressor(verbose=0, random_state=42), param_grid, n_iter=4, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
grid_cat.fit(X_train, y_train)
best_cat = grid_cat.best_estimator_
y_pred_cat = best_cat.predict(X_test)
print("Best Params:", grid_cat.best_params_)
print("Test MSE:", mean_squared_error(y_test, y_pred_cat))
print("Test R2 :", r2_score(y_test, y_pred_cat))

plt.figure(figsize=(8,3))
plt.barh(np.array(fetch_california_housing().feature_names), best_cat.feature_importances_)
plt.title("CatBoost Feature Importances")
plt.show()

# --------------------- 5. Quantile Regressor (%90 üst sınır) ---------------------
print("\n--- Quantile GBDT (0.9) ---")
model_q = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
model_q.fit(X_train, y_train)
y_pred_q = model_q.predict(X_test)
print("Test MAE (Quantile, 0.9):", mean_absolute_error(y_test, y_pred_q))
plt.figure(figsize=(7,3))
plt.plot(y_test[:50], label="Gerçek")
plt.plot(y_pred_q[:50], label="Tahmin (q=0.9)")
plt.legend()
plt.title("Quantile Regressor - 0.9")
plt.show()

# --------------------- 6. Time Series Regressor (LightGBM) ---------------------
print("\n--- TimeSeries LightGBM ---")
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for fold, (train_ix, test_ix) in enumerate(tscv.split(X)):
    X_tr, X_te = X[train_ix], X[test_ix]
    y_tr, y_te = y[train_ix], y[test_ix]
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    score = r2_score(y_te, y_pred)
    print(f"Fold {fold+1} R2: {score:.3f}")
    scores.append(score)
print("Ortalama R2 (TimeSeries):", np.mean(scores))
