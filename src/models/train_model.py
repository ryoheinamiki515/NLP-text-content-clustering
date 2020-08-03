from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import gmean, hmean


def hyperopt_data(param_space, data, num_eval):
    def objective_function(params):
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("kmeans", KMeans())
        ])
        pipe.set_params(**params)
        pipe.fit(data)
        silhouette = (silhouette_score(pipe["tfidf"].transform(data).toarray(), pipe["kmeans"].labels_)+1)/2
        davies_bouldin = 1-davies_bouldin_score(pipe["tfidf"].transform(data).toarray(), pipe["kmeans"].labels_)
        if davies_bouldin < 0:
            davies_bouldin = 0
        score = hmean([silhouette, davies_bouldin])
        return {'loss': -silhouette, 'status': STATUS_OK}
    trials = Trials()
    best_param = fmin(
        objective_function,
        param_space,
        algo=tpe.suggest,
        max_evals=num_eval,
        trials=trials
    )
    return trials, best_param

