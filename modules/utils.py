import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from keras.layers import Activation
import tensorflow as tf
from joblib import dump
import os

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

def preprocess_data(X, scaler=None, save_scaler=False, scaler_name=None):
    if scaler == None:
        scaler = StandardScaler()
        if X.ndim == 1:
            X = np.squeeze(scaler.fit_transform(X.reshape(-1, 1)))
        else:
            X = scaler.fit_transform(X)  
    else: 
        if X.ndim == 1:
            X = np.squeeze(scaler.transform(X.reshape(-1, 1)))
        else:
            X = scaler.transform(X)  
    if save_scaler:
        dump(scaler, f"{os.getcwd()}/models/{scaler_name}.joblib") 
    return X

def import_predict_dataset(url_features):
    feature_set = pd.read_csv(url_features)
    feature_set.drop('Unnamed: 0', axis=1, inplace=True)
    return feature_set

def root_mean_squared_error(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))