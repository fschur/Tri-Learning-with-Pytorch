import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch


def preprocessing_tri(train):

    # prepare labeled data
    train_all = pd.read_hdf("train_labeled.h5", "train")
    train_y = train_all['y']
    train_x = train_all.drop(['y'], axis=1)
    train_x = train_x.values
    train_y = train_y.values

    # prepare unlabeled data
    train_all_un = pd.read_hdf("train_unlabeled.h5", "train")
    train_x_un = train_all_un.values

    # scale data
    all_train = np.concatenate((train_x, train_x_un))
    scaler_x = preprocessing.StandardScaler().fit(all_train)
    scaled_x_un = scaler_x.transform(train_x_un)
    scaled_x = scaler_x.transform(train_x)

    # create test set
    test_x = pd.read_hdf("test.h5", "test")
    test_x = test_x.values
    scaled_test_x = scaler_x.transform(test_x)

    if train:
        # create train and dev set
        len_train = len(train_y)
        len_dev = int(len_train / 9)
        dev_x = scaled_x[-len_dev:]
        train_x = scaled_x[:-len_dev]
        dev_y = train_y[-len_dev:]
        train_y = train_y[:-len_dev]

        return (train_x, train_y, scaled_x_un, dev_x, dev_y, scaled_test_x)
    else:
        return (scaled_x, train_y, scaled_x_un, scaled_test_x)