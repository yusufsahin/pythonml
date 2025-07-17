
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 1. Veri setini yükle (örnek: 8 kategoriyle sınırlı)
categories = ['sci.space', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

# 2. TF-IDF Vektörleştirme
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(newsgroups.data)

# 3. Cosine mesafe matrisi
dist_matrix = cosine_distances(X_tfidf)

# 4. Linkage matrisi
Z = linkage(dist_matrix, method='ward')

# 5. Dendrogram çizimi
plt.figure(figsize=(14, 6))
dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.title("Hierarchical Clustering Dendrogram (News Articles)")
plt.xlabel("Gruplar")
plt.ylabel("Cosine Mesafe")
plt.tight_layout()
plt.show()

# 6. Küme etiketleri çıkar (örneğin 4 grup)
labels = fcluster(Z, t=4, criterion='maxclust')

# 7. İlk 10 dökümanın küme etiketlerini yazdır
print("İlk 10 dokümanın küme etiketleri:")
for i in range(10):
    print(f"{i+1}. Küme: {labels[i]}  →  Başlık: {newsgroups.filenames[i].split('/')[-1]}")
