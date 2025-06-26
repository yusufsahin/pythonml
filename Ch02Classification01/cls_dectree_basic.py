import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
# Veri setini yükle
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)
print(df.head())

# Yeni özellik: Title

df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
df["Title"] = df["Title"].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
     'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
df["Title"] = df["Title"].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})


# Eksik değerleri doldur
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna('Unknown')

# Kategorik sütunları sayısal hale getir
label_cols = ['Sex', 'Embarked', 'Title']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
# Özellik seçimi
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']
X = df[features]
y = df['Survived']

# Eğitim/test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Modeli uygula
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
#Performans
y_pred = clf.predict(X_test)
print("🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))
print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
#Cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"\n📊 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# Görselleştirme: Karar Ağacı
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=features, class_names=["Died", "Survived"], filled=True, rounded=True)
plt.title("Karar Ağacı Görselleştirmesi")
plt.show()
