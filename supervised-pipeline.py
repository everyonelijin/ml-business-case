import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from tabulate import tabulate


def split_dataset(df):
    """Split the dataset into two sub-datasets: one with many NaNs by line, another with few NaNs by line"""
    nans = df.isnull().values.astype(int)
    # Clustering on lines
    kmeans = KMeans(n_clusters=2).fit(nans)
    ind = np.array(kmeans.labels_ == 0)
    return df[ind], df[~ind], ind, ~ind


data = None


def build_fit_pipeline():
    pass


# Split du dataset
data_nan, data_full, idx_nan, idx_full = split_dataset(data)

# Dummyfication
dummyfier_nan = None
dummyfier_full = None

# Réduction de dimension
pca_nan = None
pca_full = None

# Random Forest
clf_nan = RandomForestClassifier(n_estimators=10, max_depth=4)
clf_full = RandomForestClassifier(n_estimators=10, max_depth=4)

# Cross-validation
X_nan_train, X_nan_test, y_nan_train, y_nan_test = train_test_split(data_nan, test_size=0.2, random_state=0)
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(data_full, test_size=0.2, random_state=0)

pipe_nan = Pipeline([('dummy', dummyfier_nan), ('pca', pca_nan), ('clf', clf_nan)])
pipe_full = Pipeline([('dummy', dummyfier_full), ('pca', pca_full), ('clf', clf_full)])

# Entrainement (+ CV pour les hyperparamètres)
pipe_nan.fit(X_nan_train, y_nan_train)
pipe_full.fit(X_full_train, y_full_train)

# Evaluation
confusion = metrics.confusion_matrix(y_test, y_pred)
loss = metrics.log_loss(y_test, y_pred)
print(tabulate(confusion))
