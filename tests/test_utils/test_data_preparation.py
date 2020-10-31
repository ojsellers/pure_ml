'''
Created on 31/10/2020 by Ollie

Unit tests for data_preparation.py functions
'''


import unittest
from pure_ml.utils.data_preparation import prepare_classification_data


class test_data_preparation(unittest.TestCase):

    def test_prepare_classification_data(self):

        X_train, y_train, X_test, y_test = prepare_classification_data()

        assert (len(X_test) / (len(X_train) + len(X_test)) == 0.2)
        assert (len(y_test) / (len(y_train) + len(y_test)) == 0.2)

        assert (all([i in y_test.index for i in X_test.index]) == True)
        assert (all([i in y_train.index for i in X_train.index]) == True)
        assert (all([i in y_train.index for i in y_test.index]) == False)


if __name__ == '__main__':
    unittest.main()
