import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#1. Basit Veri Seti(Reklam harcaması ve Satış)
data={
    'TV':[50,60,70,80,90,100],
    'Sales':[7,8,9,10,11,12],
}
#y=ax+b+€
df=pd.DataFrame(data)

#2. Girdiler X->TV ve Çıktılar Y-> Sales

X=df[['TV']] #Bağımsız Değişken
y=df[['Sales']] #Bağımlı Değişken

#3. Modeli Oluşturma
model=LinearRegression()
model.fit(X,y)
#4. Tahmin yap
y_pred=model.predict(X)

# 🔹 5. Değerlendirme metrik fonksiyonu
def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# 6. Sonuçları Yazdırma
print("R²:", r2_score(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("RMSE:", calculate_rmse(y, y_pred))

# 🔹 7. Modelin matematiksel formülü
print("\nModelin Denklemi:")
print("Sales =", model.coef_[0], "* TV +", model.intercept_)