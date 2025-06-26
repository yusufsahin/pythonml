import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, f1_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler

# 1. Veri yükle
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# 2. Keşifsel analiz - Sınıf dengesi, yaş dağılımı, cinsiyet farkı
fig, axs = plt.subplots(1, 3, figsize=(15,4))
sns.countplot(x="Survived", data=df, ax=axs[0])
axs[0].set_title("Sınıf Dengesi")
sns.histplot(df["Age"].dropna(), bins=30, kde=True, ax=axs[1])
axs[1].set_title("Yaş Dağılımı")
sns.countplot(x="Sex", hue="Survived", data=df, ax=axs[2])
axs[2].set_title("Cinsiyet ve Hayatta Kalma")
plt.tight_layout()
plt.show()
# 3. Eksik veriyi kontrol et ve doldur (inplace kullanmadan, future-proof!)
df["Sex"] = (df["Sex"] == "male").astype(int)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
# 4. Kategorik değişkenlerden dummy feature ekle (Embarked için)
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
# 5. Özellikler: Pclass, Sex, Age, Fare, Embarked
features = ["Pclass", "Sex", "Age", "Fare"] + [col for col in df.columns if col.startswith("Embarked_")]
X = df[features].values
y = df["Survived"].values

# 6. Ölçekleme
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 7. Eğitim/test ayrımı (stratify ile, class dengesini korur)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)


# 8. Hiperparametre analiz: C değerinin doğruluğa etkisi
c_values = np.logspace(-3, 2, 10)
cv_scores = []
for c in c_values:
    lr = LogisticRegression(C=c, max_iter=500, solver="lbfgs")
    scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())
plt.figure(figsize=(7,4))
plt.semilogx(c_values, cv_scores, marker="o")
plt.xlabel("C (Regularizasyon Katsayısı)")
plt.ylabel("CV Doğruluk (Accuracy)")
plt.title("C Değerine Göre Model Başarısı")
plt.grid(True)
plt.show()

best_c = c_values[np.argmax(cv_scores)]
print(f"En iyi C: {best_c:.3f}")

# 9. Son modeli kur, başarıya bak
best_model = LogisticRegression(C=best_c, max_iter=500, solver="lbfgs")
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# 10. Confusion Matrix, ROC Curve, Precision-Recall Curve
plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")

plt.subplot(1,3,2)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1,3,3)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.show()

# 11. Özellik katsayıları görselleştir
coefs = best_model.coef_[0]
plt.figure(figsize=(8,3))
sns.barplot(x=features, y=coefs, errorbar=None)
plt.xticks(rotation=45)
plt.title("Lojistik Regresyon Katsayıları")
plt.ylabel("Katsayı")
plt.grid(True)
plt.show()

# 12. Yaş ve Cinsiyete göre hayatta kalma oranı (extra analiz)
plt.figure(figsize=(7,5))
sns.barplot(
    x="Sex",
    y="Survived",
    data=df,
    hue=pd.cut(df["Age"], [0,12,18,50,80]),
    errorbar=None
)
plt.title("Yaş ve Cinsiyete Göre Hayatta Kalma Oranı")
plt.ylabel("Hayatta Kalma Oranı")
plt.xlabel("Cinsiyet (0=Kadın, 1=Erkek)")
plt.legend(title="Yaş Grubu")
plt.show()
