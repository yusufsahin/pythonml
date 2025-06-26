import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# 1. Başlıkları ekle ve veriyi oku
#CRIM: Kişi başı suç oranı ZN: Büyük arsa oranı INDUS: Ticari olmayan alan oranı
#CHAS: Charles Nehri’ne yakınlık (1/0) NOX: Azot oksit kirliliği RM: Ortalama oda sayısı
#AGE: Eski ev oranı DIS: İş merkezlerine uzaklık RAD: Otoyol ulaşım indeksi TAX: Emlak vergisi oranı
#PTRATIO: Öğrenci/öğretmen oranı B: Irk göstergesi LSTAT: Yoksul nüfus oranı MEDV: Ev fiyatı (1000$)


columns = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
    "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

#1 . Verileri yükleme

df= pd.read_csv("data/dataset.txt",sep=r'\s+', header=None, names=columns)
#2. Oda sayısı ile fiyat arasındaki ilişkiyi inceleme
X= df[["RM"]].values # Oda sayısı / Bağımsız değişken
y = df["MEDV"].values  # Ev fiyatı / Bağımlı değişken

#3. Modeli oluşturma ve eğitme / Doğrusal regresyon modeli
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

#4. Polinomial regresyon model özelliği ekle 2. derece
poly= PolynomialFeatures(degree=2)  # 2. derece polinomial özellikler
X_poly = poly.fit_transform(X)  # Oda sayısını polinomial özelliklere dönüştür

poly_reg= LinearRegression()
poly_reg.fit(X_poly, y)  # Polinomial regresyon modelini eğit
y_pred_poly = poly_reg.predict(X_poly)  # Polinomial modelin tahminleri

#5. Başarı metriklerini hesaplama (R2, RMSE)
print("Lineer Regresyon R2:", r2_score(y, y_pred_lin))
print("Lineer Regresyon RMSE:", np.sqrt(mean_squared_error(y, y_pred_lin)))
print("Polinomial Regresyon R2:", r2_score(y, y_pred_poly))
print("Polinomial Regresyon RMSE:", np.sqrt(mean_squared_error(y, y_pred_poly)))


#6. Model katsayılarını yazdırma
print("\nLineer Model Formülü: Fiyat = {:.2f} + {:.2f} * RM"
      .format(lin_reg.intercept_, lin_reg.coef_[0]))
print("Polinomial Model Formülü: Fiyat = {:.2f} + {:.2f} * RM + {:.2f} * RM^2"
      .format(poly_reg.intercept_, poly_reg.coef_[1], poly_reg.coef_[2]))

#7. Tahminleri ve gerçek değerleri karşılaştırma ilk 10 satır
df_tahmin = pd.DataFrame({
    "Oda Sayısı (RM)": X.flatten(),
    "Gerçek Fiyat (MEDV)": y,
    "Lineer Tahmin": y_pred_lin,
    "Polinomial Tahmin": y_pred_poly
})
print("\nİlk 10 satır için karşılaştırma:")
print(df_tahmin.head(10))
#8. Konsolda ilk 5 gözlem için karşılaştırma
print("\nİlk 5 gözlem için tahminler:")
for i in range(5):
    print(f"RM={X[i][0]:.2f} - Gerçek Fiyat: {y[i]:.2f}, Lineer Tahmin: {y_pred_lin[i]:.2f}, Polinomial Tahmin: {y_pred_poly[i]:.2f}")
#9. Grafikle göster
plt.scatter(X, y, color='red', label='Gerçek Veri', alpha=0.5)
plt.plot(X, y_pred_lin, color='green', label='Lineer Regresyon')
# X'i küçükten büyüğe sıralayarak polinomial eğriyi düzgün çiz
sort_idx = X[:,0].argsort()
plt.plot(X[sort_idx], y_pred_poly[sort_idx], color='blue', linestyle='--', label='Polinomial Regresyon (2.derece)')
plt.xlabel("Ortalama Oda Sayısı (RM)")
plt.ylabel("Ev Fiyatı (1000$)")
plt.title("Boston Housing - Oda Sayısı ile Fiyat Arası Polinomial Regresyon")
plt.legend()
plt.tight_layout()
plt.show()