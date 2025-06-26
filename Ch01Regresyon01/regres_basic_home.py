import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

#1. Basit Veri Seti (Metrekare ve Fiyat)
data = {
    'Metrekare': [50, 60, 70, 80, 90, 100],
    'Fiyat': [1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
}

df = pd.DataFrame(data)

#2. Girdiler X->Metrekare ve Çıktılar Y-> Fiyat

X= df[['Metrekare']]  # Bağımsız Değişken
y = df[['Fiyat']]      # Bağımlı Değişken

#3. Modeli Oluşturma
model = LinearRegression()
model.fit(X,y)

#4. Tahmin yap
y_pred = model.predict(X)



# 🔹 5. Değerlendirme metrik fonksiyonu



def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

# 🔹 6. Performans metriklerini yazdır
print("R²:", r2_score(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("RMSE:", calculate_rmse(y, y_pred))

# 🔹 7. Modelin matematiksel formülü
print("\nModelin Denklemi:")
print("Fiyat =", model.coef_[0][0], "* Metrekare +", model.intercept_[0])

m2_input_value = 85
# Tahmin yap
m2_input_df=pd.DataFrame([m2_input_value], columns=['Metrekare'])
tahmin_fiyat = model.predict(m2_input_df)
print(f"\n{m2_input_value} metrekarelik bir alanın tahmini fiyatı: {tahmin_fiyat[0][0]:.2f} milyon TL")