from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
#1. Veri Setini Yükle
data= fetch_california_housing()
X= data.data
y= data.target

#2.Veri setini eğitim ve test olarak ayır
# Modelin başarısını anlamak için genellikle verinin %80'ini eğitim, %20'sini test için ayırırız
X_train,X_test,y_train,y_test= train_test_split(
    X,y,test_size=0.2, random_state=42 # random_state: Sonuçlar tekrar edilebilsin diye/rastgelelik
)

#3.Karar Ağaçları Modelini Oluştur ve eğit
# Karar ağacı modelini max_depth=5 ile başlatıyoruz (ağacın derinliği - aşırı ezberlemeyi önler)
model= DecisionTreeRegressor(max_depth=5,random_state=42)
model.fit(X_train,y_train) # Modeli eğitim verisiyle eğitiyoruz
#5.Test verisi ile tahmin yap

y_pred = model.predict(X_test)
#6.Performans metriklerini hesapla ve yazdır

mse= mean_squared_error(y_test,y_pred)
r2= r2_score(y_test,y_pred)

print("Ortalama Karesel Hata ,Test Mean Squared Error (MSE):", mse) #Ne kadar düşükse o kadar iyi
print("AÇıklanan Varyans / Test R² Score:", r2)

#7.Modelin görselleştirilmesi
plt.figure(figsize=(18,6))
plot_tree(model,feature_names=data.feature_names, filled=True, max_depth=5)
plt.title('Decision Tree Regressor ilk 2 Seviyeyi Gösterir')
plt.show()