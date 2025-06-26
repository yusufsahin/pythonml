import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score

# 1. Veri setini yükle
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# 2. VERİYİ TANI
print("Veri boyutu:", X.shape)
print("Özellikler:", X.columns.tolist())
print(X.head())
print("Eksik değer var mı?:\n", X.isnull().sum())

# 3. Eksik değer temizleme (Bu veri setinde yok ama gerçek hayatta olabilir)
# Örnek: X.fillna(X.mean(), inplace=True)

# 4. Aykırı değer analizi (describe ile bakılır, burada outlier işlemi eklenmedi çünkü veri düzgün)
print("\nİstatistikler:\n", X.describe())

# 5. Feature Engineering (Özellik Mühendisliği)
X['RoomsPerPerson'] = X['AveRooms'] / X['AveOccup']
X['BedroomsPerRoom'] = X['AveBedrms'] / X['AveRooms']
X['Log_Population'] = np.log1p(X['Population'])

# 6. Özellik Ölçekleme (StandardScaler ile)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Eğer DataFrame olarak kalmasını istersen:
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 7. Eğitim/Test bölmesi
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 8. Hiperparametre araması için parametre grid'i
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train, y_train)

print("En iyi parametreler:", grid.best_params_)

# 9. Model eğitimi ve test tahmini
best_tree = grid.best_estimator_
best_tree.fit(X_train, y_train)
y_pred = best_tree.predict(X_test)

# 10. Başarı metrikleri
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Test MSE:", mse)
print("Test R2:", r2)

# 11. Görselleştirmeler (Scatter, Histogram, Ağaç, vs.)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Gerçek")
plt.ylabel("Tahmin")
plt.title("Gerçek vs Tahmin")
plt.grid(True)
plt.show()

errors = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.hist(errors, bins=30)
plt.xlabel("Tahmin Hatası")
plt.ylabel("Frekans")
plt.title("Hata Dağılımı")
plt.grid(True)
plt.show()

plt.figure(figsize=(20, 8))
plot_tree(
    best_tree,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3
)
plt.title("Karar Ağacı (İlk 3 Seviye)")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(y_test[:20].values, label="Gerçek", marker='o')
plt.plot(y_pred[:20], label="Tahmin", marker='x')
plt.title("İlk 20 Ev: Gerçek vs Tahmin")
plt.xlabel("Ev No")
plt.ylabel("Fiyat")
plt.legend()
plt.grid(True)
plt.show()
