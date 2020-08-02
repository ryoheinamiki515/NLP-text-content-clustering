from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

def hyperopt_data(param_space, data, num_eval):
    def objective_function(params):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("kmeans", KMeans())
        ])
        pipe.set_params(**params)
        pipe.fit(data)
        score = silhouette_score(data, pipe.labels_)
        return {'loss': -score, 'status': STATUS_OK}
    trials = Trials()
    best_param = fmin(
        objective_function,
        param_space,
        algo=tpe.suggest,
        max_evals=num_eval,
        trials=trials
    )
    return trials, best_param


def train_model(data, labels, n_clusters, show_results=False, changed_to_original=None):
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.2)
    vectors = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    km = KMeans(n_clusters=n_clusters)
    km.fit(vectors)
    if show_results:
        centers_ordered = km.cluster_centers_.argsort()[:, ::-1]
        for i in range(n_clusters):
            indices = np.where(km.labels_ == i)[0]
            cluster_labels = [labels[i] for i in indices]
            print(f"Cluster #{i}:", ", ".join(cluster_labels))
            prominent_words = changed_to_original.loc[df.columns[centers_ordered[i][:5]]].unique()
            print("Prominent words:", ", ".join(prominent_words))
            print("==============================\n")
    return km
