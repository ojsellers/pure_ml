'''
Created on 31/10/2020 by Ollie

Functions to prepare data sets for testing algos
'''


import pandas as pd
import numpy as np
from importlib_resources import files


def prepare_classification_data():
    '''
    Load Iris data set and prepare train and test sets
    '''
    df = pd.read_csv(files('pure_ml.data').joinpath('iris.csv'))
    return train_test_split(df.drop(columns='species'), df['species'])


def train_test_split(X, y, test_size=0.2, shuffle=True):
    '''
    Split data into train and test sets

    :param X: feature variables
                (pd.DataFrame object)
    :param y: target variable
                (pd.Series object)
    :return: X_train, y_train, X_test, y_test
    '''
    # check no instances missing from features or target
    assert len(X) == len(y)

    indices = np.arange(0, len(X))

    if shuffle:
        np.random.shuffle(indices)

    cutoff = int(len(X) - len(X) // (1 / test_size))

    train_i = indices[:cutoff]
    test_i = indices[cutoff:]

    return X.loc[train_i], y.loc[train_i], X.loc[test_i], y.loc[test_i]


def standard_scaler():
    '''
    Add function to scale normalise data by mean and variance
    '''
    pass
