import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Geniş tablo ayarı
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 10)

# 1. Diabetes veri setini yükle
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# 2. Lineer, Ridge, Lasso modelleri eğit
models = {
    "Lineer": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000)
}
tahminler = {}
for isim, model in models.items():
    model.fit(X, y)
    tahminler[isim] = model.predict(X)

# 3. Cross-validation skorları (daha gerçekçi karşılaştırma)
print("\nCross-validation R2 skorları (5-fold):")
for isim, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{isim}: {scores.mean():.3f} (std: {scores.std():.3f})")

# 4. Lasso'nun sıfırladığı feature'lar
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X, y)
coef = pd.Series(lasso.coef_, index=feature_names)
print("\nLasso tarafından sıfırlanan feature’lar:")
print(coef[coef == 0])

print("\nLasso'nun tuttuğu feature’lar ve katsayıları:")
print(coef[coef != 0].round(3))

# 5. Lasso'da farklı alpha'larla R2 ve sıfırlanan feature sayısı
alphas = [0.01, 0.1, 0.5, 1.0, 10.0]
print("\nLasso'da farklı alpha değerleri için R2 skoru ve sıfırlanan feature sayısı:")
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X, y)
    y_pred = lasso.predict(X)
    r2 = r2_score(y, y_pred)
    zero_count = np.sum(lasso.coef_ == 0)
    print(f"alpha={alpha:<5} | R2={r2:.3f} | Sıfırlanan feature: {zero_count} / {len(lasso.coef_)}")

# 6. Lasso Path: Katsayıların alpha'ya göre grafiği
alphas_path = np.logspace(-2, 1, 50)
coefs = []
for a in alphas_path:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X, y)
    coefs.append(lasso.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas_path, coefs)
plt.xscale('log')
plt.xlabel("alpha (log scale)")
plt.ylabel("Katsayılar")
plt.title("Lasso ile Katsayıların Sıfırlanması (Lasso Path)")
plt.legend(diabetes.feature_names, loc="best", fontsize=9)
plt.tight_layout()
plt.show()

# 7. Tüm modellerin katsayılarını karşılaştıran tablo
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
lin = LinearRegression()
lin.fit(X, y)
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X, y)

df_coef = pd.DataFrame({
    'Lineer': lin.coef_,
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_
}, index=diabetes.feature_names)

print("\nTüm modellerin katsayıları:\n")
print(df_coef.round(3))

# 8. En iyi alpha'yı otomatik seçmek (LassoCV)
lasso_cv = LassoCV(alphas=np.logspace(-2, 1, 50), cv=5, max_iter=10000)
lasso_cv.fit(X, y)
print("\nEn iyi alpha (LassoCV ile):", lasso_cv.alpha_)
