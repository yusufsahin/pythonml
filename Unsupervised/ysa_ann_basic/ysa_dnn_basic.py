import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# === 1. Veri ===
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign

# === 2. Split & Scale ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 3. DNN Modeli (Derin yapı) ===
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary çıkış

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === 4. EarlyStopping ===
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# === 5. Eğit ===
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, batch_size=16, callbacks=[early_stop], verbose=1)

# === 6. Değerlendirme ===
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\n" + classification_report(y_test, y_pred, target_names=data.target_names))

# === 7. Eğri Görselleştir ===
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title("DNN Kayıp Eğrisi")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
