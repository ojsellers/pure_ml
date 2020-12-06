"""
Created on 31/10/2020 by Ollie

Unit tests for data_preparation.py functions
"""


import unittest
from pure_ml.utils.data_preparation import prepare_classification_data


class test_data_preparation(unittest.TestCase):

    def test_prepare_classification_data(self):

        X_train, y_train, X_val, y_val = prepare_classification_data()

        assert (len(X_val) / (len(X_train) + len(X_val)) == 0.2)
        assert (len(y_val) / (len(y_train) + len(y_val)) == 0.2)

        assert (all([i in y_val.index for i in X_val.index]) == True)
        assert (all([i in y_train.index for i in X_train.index]) == True)
        assert (all([i not in y_train.index for i in y_val.index]) == True)


if __name__ == '__main__':
    unittest.main()
