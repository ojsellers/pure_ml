"""
Created on 31/10/2020 by Ollie

Supervised Naive Bayes classification method in pure Python
"""


import numpy as np


from pure_ml.utils.data_preparation import prepare_classification_data


class NaiveBayes:

    def fit(self, X_train, y_train):
        """
        Fit Gaussian Naive Bayes by calculating the prior, mean, standard
        deviation for each feature for each class
        """
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
        for class in self.fit_data.keys():
            pass


    def predict(self, X_val):
        # for X in X_val:
        #     for class in self.class_i.keys():
        #         prob = (self.fit[class]['prior'] *
        #             np.sum([self._gaussian_likelihood(x, )]))
        pass

if __name__ == '__main__':
    X_train, y_train, X_val, y_val = prepare_classification_data(to_numpy=True)

    model = NaiveBayes()
    model.fit(X_train, y_train)
    model.predict(X_val)
