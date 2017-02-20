import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tabulate import tabulate

from utils import *
from stacking import StackingClassifier


if __name__ == '__main__':
    df = pd.read_csv('train.csv', index_col='ID')
    print("Loading dataset... ok")

    X, y = xy_split(df)

    # Split dataset into to 2 dataframes (one with many NaNs, another with few NaNs)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=57)
    print("Creating train and validation datasets... ok")

    # Split dataset into to 2 dataframes (one with many NaNs, another with few NaNs)
    X_full, X_nan, y_full, y_nan, idx_full, idx_nan = preprocess(X_train, y_train)
    print("Splitting dataset... ok")

    # Prepare transformer pipelines and list of classifiers for first layer models in stacking
    pipe_full = Pipeline([
        ('imputer', DataFrameImputer()),
        ('dummyfier', Dummyfier()),
        ('pca', PCA(n_components=25, svd_solver='arpack')),
    ])
    pipe_nan = Pipeline([
        ('imputer', DataFrameImputer()),
        ('dummyfier', Dummyfier()),
        ('pca', PCA(n_components=25, svd_solver='arpack')),
    ])
    clfs_full = [
        ('rf25', RandomForestClassifier(n_estimators=25, max_depth=4, n_jobs=3)),
        ('rf40', RandomForestClassifier(n_estimators=40, max_depth=3, n_jobs=3)),
        ('logreg1', LogisticRegression(C=1.0)),
        ('logreg3', LogisticRegression(C=3.0)),
        # ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier(n_neighbors=10, n_jobs=3))
    ]
    clfs_nan = [
        ('rf25', RandomForestClassifier(n_estimators=25, max_depth=4, n_jobs=3)),
        ('rf40', RandomForestClassifier(n_estimators=40, max_depth=3, n_jobs=3)),
        ('logreg1', LogisticRegression(C=1.0)),
        ('logreg3', LogisticRegression(C=3.0)),
        # ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier(n_neighbors=10, n_jobs=3))
    ]

    stack_full = StackingClassifier(clfs_full, LogisticRegression(), transformer=pipe_full)
    stack_nan = StackingClassifier(clfs_nan, LogisticRegression(), transformer=pipe_nan)

    stack_full.fit(X_full, y_full, True)
    print("Fitting Full Model... ok")
    stack_nan.fit(X_nan, y_nan, True)
    print("Fitting NaN Model... ok")

    # Validation set predictions
    Xv_full, Xv_nan, yv_full, yv_nan, idxv_full, idxv_nan = preprocess(X_val, y_val)
    preds_full = stack_full.predict(Xv_full)
    probas_full = stack_full.predict_proba(Xv_full)
    preds_nan = stack_nan.predict(Xv_nan)
    probas_nan = stack_nan.predict_proba(Xv_nan)
    y_pred = reconstruct([preds_full, preds_nan], [idxv_full, idxv_nan])
    y_probas = reconstruct([probas_full, probas_nan], [idxv_full, idxv_nan])

    # Evaluation
    confusion = metrics.confusion_matrix(y_val, y_pred)
    loss = metrics.log_loss(y_val, y_probas)
    print("Confusion matrix")
    print(tabulate(confusion, tablefmt="fancy_grid"))
    print("Log-loss: {:0.4f}".format(loss))

    # Prediction
    X_test = pd.read_csv('test.csv')
    X_test_ID = X_test.ID.copy()
    X_test.drop(['ID'], axis=1, inplace=True)
    Xt_full, Xt_nan, _, _, idxt_full, idxt_nan = preprocess(X_test)
    probas_full = stack_full.predict_proba(Xt_full)
    probas_nan = stack_nan.predict_proba(Xt_nan)
    y_probas = reconstruct([probas_full, probas_nan], [idxt_full, idxt_nan])
    y_probas_df = pd.DataFrame({'ID': X_test_ID, 'PredictedProb': y_probas})
    y_probas_df.to_csv('submission_stacking.csv', index=None)
