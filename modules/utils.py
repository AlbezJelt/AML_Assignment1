import pandas as pd

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