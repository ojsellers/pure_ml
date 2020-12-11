"""
Created on 06/12/2020 by Ollie

Script to provide scoring functions for use in analysis of algorithms

"""


import numpy as np


def accuracy(y_pred, y_true):
    """
    This function computes standard accuracy of classification. Columns handed 
    to this function must be of the same type e.g. both should be numerical 

    """
    assert np.shape(y_true) == np.shape(y_pred), "Shapes must match"
    
    return np.sum(y_true == y_pred) / len(y_true)
