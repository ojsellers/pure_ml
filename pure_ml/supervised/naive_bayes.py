"""
Created on 31/10/2020 by Ollie

Supervised Naive Bayes classification method in pure Python
"""


import numpy as np


from pure_ml.utils.data_preparation import prepare_classification_data
from pure_ml.utils.scoring import accuracy


class NaiveBayes:

    def fit(self, X_train, y_train):
        """
        Fit Gaussian Naive Bayes by calculating the prior, mean, standard
        deviation for each feature for each class

        :param X_train: training data features
                    (np 2D array)
        :param y_train: target values of type int or string
                    (np 1D array)
        """
        self.X_train = X_train
        self.fit_data = {}

        for class_ in np.unique(y_train):
            i = (y_train == class_).nonzero()[0]

            prior = len(y_train[i]) / len(y_train)
            means = [np.mean(X_train[i, col]) for col in range(len(X_train[0]))]
            stds = [np.std(X_train[i, col]) for col in range(len(X_train[0]))]

            self.fit_data[class_] = {'prior': prior, 'means': means, 'stds': stds}


    def _gaussian_likelihood(self, x, mean, std):
        """
        Calculating probability of data point x belonging to class represented
        by the particular mean and standard deviation using the Gaussian
        probability function
        """
        coeff = 1 / (std * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((x - mean) / std)**2
        return coeff * np.exp(exponent)


    def _get_prediction(self, X):
        """
        Get prediction for an individual incidence of data
        """
        probs = []

        for class_ in self.fit_data.keys():
            prob_ = 0 

            for col in range(len(X)):
                prob_ += (self.fit_data[class_]['prior'] * 
                            self._gaussian_likelihood(X[col], 
                                self.fit_data[class_]['means'][col],
                                self.fit_data[class_]['stds'][col]))

            probs.append(prob_)

        return list(self.fit_data.keys())[probs.index(max(probs))]


    def predict(self, X_val):
        """
        Return predictions for entire column of X_val data handed to function

        :param X_val: validation data with same features as training data
                    (np 2d array)
        :return: 1D column of predictions matching y_train values
                    (list)
        """
        # ensure data is same shape as training data
        assert len(X_val[0]) == len(self.X_train[0]), "Not same as training data"

        return [self._get_prediction(X) for X in X_val]


if __name__ == '__main__':
    X_train, y_train, X_val, y_val = prepare_classification_data(to_numpy=True)

    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    print(accuracy(y_pred, y_val))
