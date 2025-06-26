import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veriyi Yükle
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Özellik Mühendisliği
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                   'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
df["Title"] = df["Title"].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})


# Eksik verileri doldur
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('Unknown')

# Kategorikleri sayısala çevir
label_cols = ['Sex', 'Embarked', 'Title']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 3. Özellik ve hedef değişken
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
X = df[features]
y = df['Survived']


# 4. Eğitim/Test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. GridSearchCV ile Hiperparametre Arama
param_grid = {
    'max_depth': [3,5,7,10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid= GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("En iyi parametreler:", grid.best_params_)
print("En iyi doğruluk (cv):", grid.best_score_)

# 6. En iyi model ile test seti değerlendirmesi
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print(" Accuracy:", accuracy_score(y_test, y_pred))

# 7. Özellik Önem Skoru
feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n Özellik Önem Sıralaması:")
print(feat_imp)
# 8. Görselleştir
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title("En Önemli Özellikler")
plt.tight_layout()
plt.show()

# 9. Karar Ağacı Çizimi
plt.figure(figsize=(18, 8))
plot_tree(best_model, feature_names=features, class_names=["Died", "Survived"], filled=True)
plt.title("Karar Ağacı Görselleştirmesi")
plt.show()

