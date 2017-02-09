import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from tabulate import tabulate


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


def reconstruct(list_y, list_ind):
    n = np.sum([len(y) for y in list_y])
    y_pred = np.zeros((n,list_y[0].shape[1]))
    for y, ind in zip(list_y, list_ind):
        y_pred[ind] = y
    return y_pred


def dummify(df):
    nonnum_cols = {}

    for i in range(1, 132):
        if type(df['v' + str(i)].iloc[0]) != np.float64:
            nonnum_cols['v' + str(i)] = df['v' + str(i)].unique()

    i = 0
    for i, col in enumerate(nonnum_cols):
        if len(nonnum_cols[col]) > 1000:
            print("skipping column %s" % col)
            continue
        threshold = .01 * len(df) / len(nonnum_cols[col])
        dums = pd.get_dummies(df[col])
        frequency = dums.sum()
        a = frequency > threshold

        todrop = []
        for c in nonnum_cols[col]:
            if type(c) == float and np.isnan(c):
                pass
            elif not a[c]:
                todrop.append(c)
        dums.drop(todrop, axis=1, inplace=True)
        if i == 0:
            dfnew = dums
        else:
            dfnew = pd.concat([dfnew, dums], axis=1)

    todrop = []
    for col in nonnum_cols:
        # if col != "v22":
        todrop.append(col)
    df.drop(todrop, axis=1, inplace=True)
    return pd.concat([df, dfnew], axis=1)

   
def manage_nan(df, prop_max=0.95):
   
   ind_keep = np.array(df.isnull().sum(axis=0) / df.shape[0] < prop_max)
   df_no_nan = df.iloc[:, ind_keep]

   col_float = df_no_nan.columns[df_no_nan.dtypes == "float64"].values

   for col in col_float:
       median = np.median(df_no_nan[col].dropna().values)
       df_no_nan[col].fillna(median, inplace=True)

   col_int = df_no_nan.columns[df_no_nan.dtypes == "int64"].values

   for col in col_int:
       impute = Imputer(strategy="most_frequent", copy=False)
       impute.fit_transform(df_no_nan[col].dropna())

   col_cat = df_no_nan.columns[df_no_nan.dtypes == "object"].values

   for col in col_cat:
       tab = df_no_nan[col].dropna().tolist()
       mc = max(set(tab), key=tab.count)
       df_no_nan[col].fillna(mc, inplace=True)
   
   return df_no_nan


def preprocess(data_nan, data_full, y, idx_nan, idx_full):
    # Gestion des NaN
    data_nan = data_nan.fillna(value=-999)
    data_full = data_full.fillna(value=-999)
    print("NaN management...ok")

    # Get X and y
    X_nan = data_nan
    y_nan = y[idx_nan]
    X_full = data_full
    y_full = y[idx_full]

    return X_nan, X_full, y_nan, y_full


def prediction(pipe_nan, pipe_full, X_nan, X_full, idx_nan, idx_full):
    y_pred_nan = pipe_nan.predict_proba(X_nan)
    print (y_pred_nan)
    y_pred_full = pipe_full.predict_proba(X_full)
    y_pred = reconstruct([y_pred_nan, y_pred_full], [idx_nan, idx_full])
    return y_pred


if __name__ == '__main__':
    df = pd.read_csv('train.csv', index_col='ID')
    df = dummify(df)
    print("Dummyfication...ok")

    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=57)

    # Split du dataset
    list_df, list_ind = split_dataset(X_train)
    data_nan, data_full, idx_nan, idx_full = list_df[0], list_df[1], list_ind[0], list_ind[1]
    print("Split...ok")

    # Preprocess data
    X_nan, X_full, y_nan, y_full = preprocess(data_nan, data_full, y_train, idx_nan, idx_full)


    # Réduction de dimension
    pca_nan = PCA(n_components=25)
    pca_full = PCA(n_components=25)

    # Random Forest
    clf_nan = RandomForestClassifier(n_estimators=40, max_depth=4)
    clf_full = RandomForestClassifier(n_estimators=40, max_depth=4)

    # Cross-validation
    # X_nan_train, X_nan_test, y_nan_train, y_nan_test = train_test_split(X_nan, y_nan, test_size=0.2, random_state=42)
    # X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    pipe_nan = Pipeline([('pca', pca_nan), ('clf', clf_nan)])
    pipe_full = Pipeline([('pca', pca_full), ('clf', clf_full)])

    # Entrainement (+ CV pour les hyperparamètres)
    pipe_nan.fit(X_nan, y_nan)
    # print("Fitting NaN Model...ok")
    pipe_full.fit(X_full, y_full)
    # print("Fitting Full Model...ok")

    # Evaluation
    # Prepare validation dataset
    test_df, test_ind = split_dataset(X_test)
    data_test_nan, data_test_full, idx_test_nan, idx_test_full = test_df[0], test_df[1], test_ind[0], test_ind[1]
    # Preprocess data
    X_test_nan, X_test_full, y_test_nan, y_test_full = preprocess(data_test_nan, data_test_full, y_test, idx_test_nan, idx_test_full)

    y_proba = prediction(pipe_nan, pipe_full, X_test_nan, X_test_full, idx_test_nan, idx_test_full)
    y_proba = pd.DataFrame(np.concatenate((y_proba,y_test.values.reshape(y_test.values.shape[0],1)), axis=1))
    y_proba.to_csv('validation.csv')

    # # Prediction
    # df_test = pd.read_csv('test.csv', index_col='ID')
    # test_df, test_ind = split_dataset(X_test)
    # data_test_nan, data_test_full, idx_test_nan, idx_test_full = test_df[0], test_df[1], test_ind[0], test_ind[1]
    # # Get X and y
    # X_test_nan = data_test_nan
    # y_test_nan = y_test[idx_test_nan]
    # X_test_full = data_test_full
    # y_test_full = y_test[idx_test_full]

    # y_pred = prediction(pipe_nan, pipe_full, X_test_nan, X_test_full, idx_test_nan, idx_test_full)
    # y_pred.to_csv('submission.csv', headers=['ID','PredictedProb'])
