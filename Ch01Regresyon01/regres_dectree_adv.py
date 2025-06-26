from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree

RANDOM_STATE = 42

#1. Veri setini yükle
data= fetch_california_housing()
X=data.data
y=data.target
#2. Veriyi tanı - Boyut kontrolü, başlılar ,veri tipi
print("Veri seti boyutu (gözlem, özellik):", X.shape)
print("Özellikler:", data.feature_names)
print("İlk 5 örnek (X):", X[:5])
print("İlk 5 hedef (y):", y[:5])

#3.EĞİTİM/TEST AYRIMI: Model başarısını tarafsız ölçmek için (test_size=%20 önerilir)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
#4.KARAR AĞACI REGRESYONU: Model kur ve hiperparametreleri açıkla
# - max_depth: Ağacın maksimum derinliği. Düşük değer aşırı basitleştirir (underfit), çok büyük değer aşırı öğrenir (overfit).
# - min_samples_split: Bir dalın ayrılabilmesi için gereken minimum örnek sayısı. Büyük değerler daha sade ağaç oluşturur.
# - min_samples_leaf: Bir yaprakta bulunması gereken minimum örnek sayısı. Overfit'i önler.
# - max_features: Her split'te göz önüne alınacak maksimum özellik sayısı. None: tüm özellikler, sqrt/log2: rastgele kısıtlar.

param_grid = {
    "max_depth": [3, 5, 7, 10, 15, None],           # Derinliği test etmek için farklı değerler
    "min_samples_split": [2, 5, 10, 20],            # Dal bölünmesi için minimum örnek sayısı
    "min_samples_leaf": [1, 2, 4, 8],               # Yaprakta minimum örnek sayısı
    "max_features": [None, "sqrt", "log2"]          # Özellik sayısının sınırlandırılması
}

#5. Hiperparametre optimizasyonu: GridSearchCV ile tüm parametre kombinasyonlarını çapraz doğrulamayla test et
# - scoring="neg_mean_squared_error": Küçük MSE daha iyi sonuç demek
# - cv=3: 3 katlı çapraz doğrulama (daha fazla olursa daha sağlam, ama daha yavaş)
# - n_jobs=-1: Tüm CPU çekirdeklerini kullanır
# - verbose=2: Eğitim sürecinde detaylı çıktı verir
grid = GridSearchCV(
    DecisionTreeRegressor(random_state=RANDOM_STATE),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train, y_train)  # Sadece eğitim verisiyle fit edilir!

print("\nEn iyi parametreler (GridSearchCV ile otomatik seçildi):", grid.best_params_)

# 6. En iyi parametrelerle modeli tekrar eğit (best practice: overfit riskini azaltır)
best_tree = grid.best_estimator_
best_tree.fit(X_train, y_train)

#7. Test veri setinde tahmin yap (gerçek performans ölçümü burada!)
y_pred = best_tree.predict(X_test)

#8.Başarı değerlendirme
# - mean_squared_error: Küçükse iyi, büyükse kötü.
# - r2_score: 0-1 arası değer alır, 1 mükemmel, 0 model hiçbir şey öğrenmemiş demektir.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nTest MSE (Ortalama Karesel Hata):", mse)
print("Test R² (Açıklanan Varyans Skoru):", r2)

# 9. Tahmin vs Gerçek değerlerin scatter plot'u
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3, color="seagreen")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Gerçek Ev Fiyatı ($100,000)")
plt.ylabel("Tahmin Edilen Ev Fiyatı ($100,000)")
plt.title("Karar Ağacı: Gerçek vs Tahmin Değerleri")
plt.grid(True)
plt.show()

# 10. Hata dağılımı histogramı
errors = y_test - y_pred
plt.figure(figsize=(8, 4))
plt.hist(errors, bins=30, color="lightblue", edgecolor="black")
plt.xlabel("Tahmin Hatası")
plt.ylabel("Frekans")
plt.title("Karar Ağacı Tahmin Hata Dağılımı")
plt.grid(True)
plt.show()

# 11. Karar Ağacını Görselleştir (İlk 3 Seviye) -- Yorumu kolaylaştırır
plt.figure(figsize=(20, 8))
plot_tree(
    best_tree,
    feature_names=data.feature_names,
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3  # Büyük ağaçlarda 3 seviye ile sınırla, okunabilirlik için
)
plt.title("Karar Ağacı (İlk 3 Seviye)")
plt.show()

# 15. İlk 20 tahmin ve gerçek değeri yan yana çiz (trend doğru yakalanıyor mu?)
plt.figure(figsize=(12, 5))
plt.plot(y_test[:20], label="Gerçek", marker='o')
plt.plot(y_pred[:20], label="Tahmin", marker='x')
plt.title("Karar Ağacı - İlk 20 Ev: Gerçek vs Tahmin")
plt.xlabel("Ev No")
plt.ylabel("Fiyat ($100,000)")
plt.legend()
plt.grid(True)
plt.show()
