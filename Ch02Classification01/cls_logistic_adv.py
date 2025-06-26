import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import shap
import matplotlib.pyplot as plt

#Veri setini yükleme
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
#Yeni özelikler ekleme
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
     'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

df["Title"] = df["Title"].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
#Eksik değerleri doldurma
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('Unknown')

#Kategorik değişkenleri dönüştürme
label_cols = ['Sex', 'Embarked', 'Title']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
# Özellikler ve hedef değişken
# Özellikler ve hedef değişken
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
X = df[features]
y = df['Survived']

# Eğitim/test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model uygula Lojistik Regresyon modeli
model = LogisticRegression(max_iter=300)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"📊 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

model.fit(X_train, y_train)

# Test değerlendirme
y_pred = model.predict(X_test)
print("\n🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🔹 Accuracy Score:", accuracy_score(y_test, y_pred))


# SHAP: yorumlanabilirlik (uyumlu masker ile)
masker = shap.maskers.Independent(X_train)
explainer = shap.LinearExplainer(model, masker)
shap_values = explainer.shap_values(X_test)


# SHAP summary plot
shap.summary_plot(shap_values, X_test, feature_names=features)
