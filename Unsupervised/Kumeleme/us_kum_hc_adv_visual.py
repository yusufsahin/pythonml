
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from wordcloud import WordCloud

# 1. Veri setini yükle (sınırlı kategorilerle)
categories = ['sci.space', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

# 2. TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tfidf = vectorizer.fit_transform(newsgroups.data)

# 3. Cosine mesafe matrisi
dist_matrix = cosine_distances(X_tfidf)

# 4. Linkage matrisi
Z = linkage(dist_matrix, method='ward')

# 5. Dendrogram (etiketli ve renklendirilmiş)
labels_text = [f"{i+1}. {newsgroups.target_names[newsgroups.target[i]]}" for i in range(len(newsgroups.data))]

plt.figure(figsize=(14, 6))
dendrogram(Z,
           labels=labels_text,
           truncate_mode='lastp',
           p=30,
           leaf_rotation=90.,
           leaf_font_size=9.,
           show_contracted=True,
           color_threshold=20)
plt.title("Hierarchical Clustering Dendrogram (Etiketli)")
plt.xlabel("Belge")
plt.ylabel("Cosine Mesafe")
plt.tight_layout()
plt.show()

# 6. Küme etiketleri çıkar (örnek: 4 küme)
labels = fcluster(Z, t=4, criterion='maxclust')

# 7. WordCloud ile her kümenin özetini çıkar
for cluster_id in range(1, 5):
    docs_in_cluster = [newsgroups.data[i] for i in range(len(labels)) if labels[i] == cluster_id]
    text = " ".join(docs_in_cluster)
    wordcloud = WordCloud(background_color='white', max_words=100).generate(text)

    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Küme {cluster_id} – WordCloud")
    plt.tight_layout()
    plt.show()

# 8. PCA ile 2D görselleştirme
X_reduced = PCA(n_components=2).fit_transform(X_tfidf.toarray())

plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='Set2')
plt.title("Hierarchical Clustering (2D PCA Görselleştirme)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Küme ID")
plt.tight_layout()
plt.show()
