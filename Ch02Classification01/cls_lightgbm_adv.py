import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Veri seti ===
X, y = load_iris(return_X_y=True)

# === 2. Eğitim/test bölmesi ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# === 3. Model ve parametre alanı ===
model = LGBMClassifier(objective='multiclass', num_class=3, verbose=-1, random_state=42)

param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'num_leaves': [15, 31, 63],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# === 4. Eğit ve en iyi modeli bul ===
search.fit(X_train, y_train)
print("✅ En iyi parametreler:", search.best_params_)

best_model = search.best_estimator_

# === 5. Test sonuçları ===
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === 6. Özellik önemi ===
importances = best_model.feature_importances_
feature_names = load_iris().feature_names

sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
