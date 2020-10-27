import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras import backend as K

def merge_training_dataset(feature_set, label_set):
    merged = pd.merge(feature_set, label_set, on='id')
    return merged

def import_training_dataset(url_features, url_labels):
    feature_set = pd.read_csv(url_features).rename(columns={'Unnamed: 0':'id'})
    label_set = pd.read_csv(url_labels).rename(columns={'Unnamed: 0':'id'})
    train_set = merge_training_dataset(feature_set, label_set)
    train_set.drop('id', axis=1, inplace=True)
    return train_set

def prepare_data(train_set, test_set):
    Y_train = train_set.copy().pop("price").to_numpy(dtype=np.float32)
    Y_test = test_set.copy().pop("price").to_numpy(dtype=np.float32)
    return train_set.copy().to_numpy(dtype=np.float32), Y_train, test_set.copy().to_numpy(dtype=np.float32), Y_test

def preprocess_data(X : np.ndarray, scaler=None):
    if not scaler:
        scaler = StandardScaler()
    if X.ndim == 1:
        X = np.squeeze(scaler.fit_transform(X.reshape(-1, 1)))
    else:
        X = scaler.fit_transform(X)      
    return X

def import_predict_dataset(url_features):
    feature_set = pd.read_csv(url_features)
    feature_set.drop('Unnamed: 0', axis=1, inplace=True)
    return feature_set

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 