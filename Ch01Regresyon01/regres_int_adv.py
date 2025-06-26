import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#1. Veri yükleme
df=pd.read_csv("data/Advertising.csv", index_col=0)

#2. Girdi (X) ve Hedef (Y) değişkenlerini ayırma
X= df[['TV', 'Radio', 'Newspaper']]  # Bağımsız Değişkenler
y = df['Sales']  # Bağımlı Değişken

#3. Eğitim ve Test setlerine ayırma
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, random_state=42)

#4. Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

#5. Tahmin yapma ve değerlendirme/metrik hesaplama
y_pred = model.predict(X_test)
rmse=np.sqrt(mean_squared_error(y_test, y_pred)) #RMSE hesaplama her sürümde çalıştır
print(f"R2 Score {r2_score(y_test, y_pred):.3f}")
print(f"Mean Squared Error {mean_squared_error(y_test, y_pred):.3f}")
print(f"Root Mean Squared Error {rmse:.3f}")

# 6. Katsayılar
print("\nKatsayılar:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.2f}")
print(f"Sabit (intercept): {model.intercept_:.2f}")

#7.Görselleştirme
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],'r--')
plt.xlabel('Gerçek Satış')
plt.ylabel('Tahmin Edilen Satış')
plt.title('Gerçek vs Tahmin Edilen Satış')
plt.tight_layout()
plt.show()