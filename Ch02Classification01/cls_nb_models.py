import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import (
    MultinomialNB, BernoulliNB, ComplementNB,
    GaussianNB, CategoricalNB
)
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, classification_report

# Stopwords indir
nltk.download('stopwords')

# === 1. VERİYİ YÜKLE ===
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                 sep='\t', names=['label', 'message'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# === 2. TF-IDF Vektörleştirme ===
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 3. MODELLERİ TANIMLA ===
models = {
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
}

# === 4. GaussianNB için sparse -> dense dönüşüm ===
X_train_dense = X_train_vec.toarray()
X_test_dense = X_test_vec.toarray()
models["GaussianNB"] = GaussianNB()

# === 5. CategoricalNB için binning + encoding gerekiyor ===
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
X_train_cat = discretizer.fit_transform(X_train_dense)
X_test_cat = discretizer.transform(X_test_dense)
models["CategoricalNB"] = CategoricalNB()

# === 6. EĞİT VE TEST ET ===
results = []

for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    if name in ["GaussianNB"]:
        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)
    elif name in ["CategoricalNB"]:
        model.fit(X_train_cat, y_train)
        y_pred = model.predict(X_test_cat)
    else:
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    results.append((name, acc))

# === 7. GÖRSEL SONUÇ ===
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
sns.barplot(x="Accuracy", y="Model", data=results_df.sort_values("Accuracy", ascending=False))
plt.title("Naive Bayes Model Comparison (TF-IDF)")
plt.xlim(0.8, 1.0)
plt.show()
