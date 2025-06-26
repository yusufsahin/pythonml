import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Veri setini yükle
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# 2. İlk inceleme
print(df.head())
print("\nEksik değerler:\n", df.isnull().sum())

# 3. Eksik değer temizleme
# 3. Eksik değer temizleme
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop('Cabin', axis=1, inplace=True)

# 4. Kategorik verileri sayısal hale getir
# Cinsiyet: 'male'->0, 'female'->1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# Embarked: One-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 5. Feature Engineering (yeni özellikler)
# Aile boyutu (kardeş/eş + ebeveyn/çocuk + 1 (kendisi))
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Yalnız mı? (binary feature)
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
# Yaş*Pclass etkileşimi
df['Age*Class'] = df['Age'] * df['Pclass']
# 6. Kullanılacak özellikler
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_Q', 'Embarked_S', 'FamilySize', 'IsAlone', 'Age*Class']
X = df[features]
y = df['Survived']

# 7. Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Eğitim/test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 9. Karar ağacı ve GridSearch
param_grid = {
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid=param_grid,
                    cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("En iyi parametreler:", grid.best_params_)
# 10. Test setinde başarı
y_pred = grid.best_estimator_.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 11. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# 12. Karar Ağacı Görselleştir (ilk 3 seviye)
plt.figure(figsize=(20, 10))
plot_tree(grid.best_estimator_,
          feature_names=features,
          class_names=["Not Survived", "Survived"],
          filled=True, rounded=True, max_depth=3)
plt.title("Karar Ağacı (İlk 3 Seviye)")
plt.show()