import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === 1. Veri Seti ===
from tensorflow.keras.datasets import boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# === 2. Ölçekleme ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 3. Model Oluşturma ===
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))  # Regresyon için tek çıktı

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# === 4. EarlyStopping ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === 5. Model Eğitimi ===
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=200,
                    batch_size=16,
                    callbacks=[early_stop],
                    verbose=1)

# === 6. Tahmin ve Değerlendirme ===
y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# === 7. Öğrenme Eğrisi ===
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title("Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, c='blue', alpha=0.7, edgecolors='k')
plt.plot([0, 50], [0, 50], '--r')
plt.xlabel("Gerçek Fiyat ($1000)")
plt.ylabel("Tahmin Fiyat ($1000)")
plt.title("Gerçek vs Tahmin")
plt.tight_layout()
plt.grid()
plt.show()

# === 8. Modeli Kaydet ===
model.save("house_price_model.h5")
print("\nModel kaydedildi: house_price_model.h5")

# === 9. Geri Yükleme Testi (İsteğe Bağlı) ===
# model = load_model("house_price_model.h5")
