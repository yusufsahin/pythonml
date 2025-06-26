import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Veri setini yükle
X, y = load_iris(return_X_y=True)

# 2. Eğitim/Test ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. LightGBM Dataset formatına dönüştür
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 4. Parametreleri belirle
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'verbose': -1
}

# 5. Modeli eğit
model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=100,
    early_stopping_rounds=10,
    verbose_eval=False
)

# 6. Tahmin ve değerlendirme
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
