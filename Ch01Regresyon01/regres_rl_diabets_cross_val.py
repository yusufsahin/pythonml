from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

# 1. Veri setini yükle
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# 2. Modelleri tanımla
models = {
    "Lineer": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000)
}

print("Cross-validation R2 skorları (5-fold):")
for isim, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{isim}: {scores.mean():.3f} (std: {scores.std():.3f})")

# 3. Lasso'nun hangi feature'ları sıfırladığını bul
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X, y)
coef = pd.Series(lasso.coef_, index=feature_names)
print("\nLasso tarafından sıfırlanan (önemsiz bulunan) feature’lar:")
print(coef[coef == 0])

print("\nLasso'nun kullandığı (önemsiz bulmadığı) feature’lar ve katsayıları:")
print(coef[coef != 0].round(3))