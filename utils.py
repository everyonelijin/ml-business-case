import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer


def copy_estimator(est):
    """Util function to duplicate a sklearn.base.BaseEstimator object"""
    return est.__class__(**est.get_params(deep=True))


class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.
        Columns of dtype object are imputed with the most frequent value in column.
        Columns of other types are imputed with mean of column.
        """
        pass

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


class Dummyfier(TransformerMixin):
    def __init__(self):
        """Transform categorical features into one-hot encoded columns."""
        pass

    def fit(self, X, y=None):
        # Find categorical features
        cat_cols = [c for c in X.columns if X[c].dtype == np.dtype('O')]
        self.features_ = dict(zip(cat_cols, [X[c].unique() for c in cat_cols]))

        # Deal with too numerous values feature (e.g. v22)
        self.to_drop = []
        for c, values in self.features_.items():
            if values.shape[0] > 1000:
                self.to_drop.append(c)
        for c in self.to_drop:
            del self.features_[c]

        # Fit one LabelBinarizer per categorical feature
        self.binarizers_ = {}
        for c in self.features_:
            self.binarizers_[c] = LabelBinarizer(sparse_output=False).fit(X[c])

        return self

    def transform(self, X, y=None):
        # Compute one-hot encoded matrix for each categorical feature
        X_new = []
        for c, b in self.binarizers_.items():
            X_b = pd.DataFrame(b.transform(X[c]), columns=["%s_%s" % (c, v) for v in b.classes_], index=X.index)
            X_new.append(X_b)

        # Drop categorical features
        X_ = X.drop(list(self.features_.keys()) + self.to_drop, axis=1)

        return pd.concat([X_] + X_new, axis=1)


def split_dataset(df, K=2):
    nans = df.isnull().values.astype(int)

    # Clustering on lines
    kmeans = KMeans(n_clusters=K).fit(nans)

    # Get datasets & corresponding indexes
    list_df = []
    list_ind = []

    for k in range(K):
        ind = np.array(kmeans.labels_ == k)
        list_df.append(df[ind])
        list_ind.append(ind)

    return list_df, list_ind


def prepare_dataframes(df):
    dfs, idxs = split_dataset(df)
    df_1, df_2, idx_1, idx_2 = dfs[0], dfs[1], idxs[0], idxs[1]

    # Check which one contains most NaNs
    if any([all(c) for c in df_1.isnull().values.T]):
        df_nan, idx_nan, df_full, idx_full = df_1, idx_1, df_2, idx_2
    else:
        df_nan, idx_nan, df_full, idx_full = df_2, idx_2, df_1, idx_1

    return df_full, df_nan, idx_full, idx_nan


def drop_na(df, threshold=0.95):
    # Compute NaN rate per column
    nan_rates = df.isnull().values.astype(np.float32).mean(axis=0)

    # Drop columns with NaN rate greater than threshold
    to_drop = [c for (i, c) in enumerate(df) if nan_rates[i] > threshold]
    df = df.drop(to_drop, axis=1, inplace=False)

    return df


def xy_split(df, target='target'):
    return df.drop([target], axis=1), df[target]


def preprocess(X, y=None):
    X_full, X_nan, idx_full, idx_nan = prepare_dataframes(X)
    X_full = drop_na(X_full)
    X_nan = drop_na(X_nan)
    if y is not None:
        y_full = y[idx_full]
        y_nan = y[idx_nan]
    else:
        y_full, y_nan = None, None
    return X_full, X_nan, y_full, y_nan, idx_full, idx_nan


def reconstruct(y_list, idx_list):
    p = y_list[0].ndim
    assert all([y.ndim == p for y in y_list]), "Arrays in list must have the same ndim."

    n = sum([y.shape[0] for y in y_list])
    if p == 1:
        out = np.zeros((n, ))
        for idx, y in zip(idx_list, y_list):
            out[idx] = y
        return out
    else:
        p = y_list[0].shape[1]
        out = np.zeros((n, p))
        for idx, y in zip(idx_list, y_list):
            out[idx, :] = y
        return out
