import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils import copy_estimator


class StackingClassifier():
    def __init__(self, probas_clfs, final_clf, transformer=None):
        """
        Args:
            probas_clfs (list of tuples): list of estimators to predict the probabilities, it takes to form of a list of tuple `[('est1_name', est1), ('est2_name', est2), ...]`
            final_clf: the final classifier, trained over the intermediate probabilities
            transformer (sklearn.base.TransformerMixin): object that processes transformations over the dataset (dummyfication, PCA, etc.)
        """
        self.transformer = transformer
        # we need an independent set of estimators for each fold of the CV for the 1st level model
        self.probas_clfs = [(name, copy_estimator(clf)) for name, clf in probas_clfs]  # TODO: maybe copy_estimator is not useful anymore
        self.final_clf = final_clf
        self._transformer_fitted = transformer is None
        self._layer_fitted = False
        self._layer_probas = None
        self._final_fitted = False

    def _fit_transformer(self, X, y=None):
        if self.transformer is not None:
            self.transformer.fit(X, y)
            self._transformer_fitted = True

    def _transform(self, X):
        if not self._transformer_fitted:
            raise Exception("Transformer not fitted.")
        return X if self.transformer is None else self.transformer.transform(X)

    def _fit_layer(self, X, y, verbose=False):
        probas = np.zeros((y.shape[0], len(self.probas_clfs)))
        for j, (clf_name, clf) in enumerate(self.probas_clfs):
            if verbose:
                print("Estimator '%s'" % (clf_name))
            clf.fit(self._transform(X), y)
            probas[:, j] = clf.predict_proba(self._transform(X))[:, 1]
        self._layer_fitted = True
        return probas

    def _predict_layer_probas(self, X):
        if not self._layer_fitted:
            raise Exception("Intermediate estimators not fitted.")
        probas = np.empty((X.shape[0], len(self.probas_clfs)))
        for j, (_, c) in enumerate(self.probas_clfs):
            probas[:, j] = c.predict_proba(self._transform(X))[:, 1]
        return probas

    def _fit_final(self, probas, y, verbose=False):
        if not self._layer_fitted:
            raise Exception("Intermediate estimators not fitted.")
        self.final_clf.fit(probas, y)
        if verbose:
            print("Final classifier fitted")
        self._final_fitted = True

    def fit(self, X, y, verbose=False):
        self._fit_transformer(X, y)
        self._layer_probas = self._fit_layer(X, y, verbose)
        self._fit_final(self._layer_probas, y, verbose)

    def predict_proba(self, X):
        if not self._layer_fitted:
            raise Exception("Intermediate estimators not fitted.")
        if not self._final_fitted:
            raise Exception("Final classifier not fitted.")
        probas = self._predict_layer_probas(X)
        return self.final_clf.predict_proba(probas)[:, 1]

    def predict(self, X, threshold=0.5):
        if not self._layer_fitted:
            raise Exception("Intermediate estimators not fitted.")
        if not self._final_fitted:
            raise Exception("Final classifier not fitted.")
        return (self.predict_proba(X) >= threshold).astype(int)
