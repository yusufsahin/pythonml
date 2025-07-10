import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# === 1. VERİ SETİ ===
df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")

# === 2. KATEGORİK DEĞİŞKENLERİ DÖNÜŞTÜR ===
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# === 3. SPLIT & SCALE ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 4. DNN MODELİ ===
from tensorflow.keras.models import load_model

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

# === 5. EĞİTİM SONU DEĞERLENDİRME ===
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# === 6. CONFUSION MATRIX ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === 7. SHAP EXPLAINER (Sınırlı veriyle) ===
explainer = shap.Explainer(model, X_test[:100])
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], feature_names=X.columns)

# === 8. RANDOM FOREST + GRIDSEARCHCV ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}
gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
gs.fit(X_train, y_train)
print("\nBest Accuracy (RandomForest):", gs.best_score_)
print("Best Params:", gs.best_params_)

# === 9. CROSS VALIDATION (GBM) ===
gbm = GradientBoostingClassifier()
scores = cross_val_score(gbm, X, y, cv=5, scoring='accuracy')
print("GBM 5-Fold Accuracy: %.2f%%" % (scores.mean() * 100))

# === 10. MODEL KAYDETME ===
model.save("obesity_dnn_model.h5")
joblib.dump(gs.best_estimator_, "obesity_rf_model.pkl")
print("Model saved.")
