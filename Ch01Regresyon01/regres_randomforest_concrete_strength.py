# 1. Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# 2. Veri Setini Yükle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
df = pd.read_excel(url, engine='xlrd')  # engine parametresi ile hata önlenir
# 3. Kolon Temizliği ve Yeniden Adlandırma
df.columns = [col.strip() for col in df.columns]  # boşlukları sil
df.rename(columns={"Concrete compressive strength(MPa, megapascals)": "Strength"}, inplace=True)
# 4. İlk İnceleme
print("🔍 İlk 5 Satır:\n", df.head())
print("\n📐 Veri Boyutu:", df.shape)
print("\n🧼 Eksik Veri:\n", df.isnull().sum())
# 5. Korelasyon Matrisi
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.show()
# 6. Giriş/Çıkış Değişkenleri
X = df.drop("Strength", axis=1)
y = df["Strength"]

# 7. Eğitim/Test Ayırımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 8. Model Oluşturma ve Eğitme
model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_train, y_train)
# 9. Tahmin
y_pred = model.predict(X_test)

# 10. Performans Değerlendirmesi
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Model Performansı:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# 11. Özellik Önem Dereceleri
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(10, 6), title="Özellik Önem Dereceleri")
plt.tight_layout()
plt.show()

# 12. Tahmin vs Gerçek Değer
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Gerçek Dayanım (MPa)")
plt.ylabel("Tahmin Edilen Dayanım (MPa)")
plt.title("Gerçek vs Tahmin")
plt.tight_layout()
plt.show()

# 13. Modeli Kaydet (Opsiyonel)
joblib.dump(model, "rf_concrete_model.pkl")
print("Model başarıyla 'rf_concrete_model.pkl' olarak kaydedildi.")


