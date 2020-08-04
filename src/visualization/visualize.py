from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def visualize_initial(clean_data):
    plt.ion()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(clean_data)
    feature_names = np.array(vectorizer.get_feature_names())
    X_embedded = TSNE(n_components=2, perplexity=8, n_iter=5000).fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.show()
    n_clusters = int(input("Input Number of Clusters: "))
    plt.close()
    return X, X_embedded, feature_names, n_clusters


def print_results(kmeans, labels, changed_to_original, feature_names):
    centers_ordered = kmeans.cluster_centers_.argsort()[:, ::-1]
    for i in range(kmeans.n_clusters):
        indices = np.where(kmeans.labels_ == i)[0]
        cluster_labels = [labels[i] for i in indices]
        print(f"Cluster #{i}:", ", ".join(cluster_labels))
        prominent_words = changed_to_original.loc[feature_names[centers_ordered[i][:5]]].unique()
        print("Prominent words:", ", ".join(prominent_words))
        print("==============================\n")


def visualize_final(kmeans, X_embedded, labels):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=kmeans.labels_, cmap='viridis')
    for i, txt in enumerate(labels):
        ax.annotate(txt, (X_embedded[:, 0][i], X_embedded[:, 1][i]))
    plt.show()
